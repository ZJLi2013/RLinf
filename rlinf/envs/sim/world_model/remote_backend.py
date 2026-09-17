# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend that advances frames on a world model served over HTTP.

The env-facing verbs are fixed (``open_session`` / ``generate`` / ``close_session``), and
:class:`WorldModelTransport` is where the wire surface plugs in. Only the one-shot generation
endpoint is implemented here; a session-scoped rollout API expresses the same step and would
be a second transport, not a change to the backend.

A full-sequence diffusion world model keeps no history across calls, so the condition window
stays on this side and rides along with every request. That also makes a retry free: the
window only advances after a response arrives, so re-sending a step cannot double-advance the
trajectory.
"""

from __future__ import annotations

import http.client
import io
import json
import time
import urllib.request
import uuid
from typing import Any, Protocol, Sequence

import numpy as np
import torch

from rlinf.envs.sim.world_model.backend import FrameQueue

__all__ = ["RemoteWorldModelBackend"]


class WorldModelTransport(Protocol):
    """Carries the three verbs over some wire surface."""

    def open(self, session_id: str, task: Any, seed: int) -> None: ...

    def step(
        self,
        session_id: str,
        step_id: int,
        cond_frames: np.ndarray,
        actions: np.ndarray,
        seed: int,
        task: Any,
    ) -> np.ndarray:
        """Return the served frames as ``[T, H, W, 3]`` uint8."""

    def close(self, session_id: str) -> None: ...


class RemoteWorldModelBackend:
    """Advances frames on a served world model.

    Args:
        cfg: The env config. Generation geometry comes from the usual keys; everything about
            the served model lives under ``world_model.remote``.
        device: Where returned frames land.
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.chunk = cfg.chunk
        self.condition_frame_length = cfg.condition_frame_length
        self.num_frames = cfg.num_frames
        self.image_size = tuple(cfg.image_size)
        self.retain_action = cfg.get("retain_action", True)
        if self.num_frames != self.condition_frame_length + self.chunk:
            raise ValueError(
                f"num_frames must be condition_frame_length + chunk; got {self.num_frames} != "
                f"{self.condition_frame_length} + {self.chunk}"
            )

        self.remote_cfg = cfg.world_model.remote
        self.max_retries = self.remote_cfg.get("max_retries", 2)
        self.step_budget_s = self.remote_cfg.get("step_budget_s", 900)
        name = self.remote_cfg.get("transport", "videos_sync")
        transports = {
            "videos_sync": VideosSyncTransport,
            "bwm_videos_sync": BwmVideosSyncTransport,
        }
        if name not in transports:
            raise ValueError(
                f"unknown transport {name!r}; expected one of {sorted(transports)}"
            )
        self._transport: WorldModelTransport = transports[name](
            self.cfg, self.remote_cfg
        )
        self._sessions: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _to_uint8(frame: torch.Tensor) -> np.ndarray:
        """``[C, 1, H, W]`` in ``[-1, 1]`` or ``[0, 1]`` to ``[H, W, C]`` uint8."""
        img = np.transpose(frame[:, 0].detach().cpu().numpy(), (1, 2, 0))
        if img.max() <= 1.2:
            img = (img + 1.0) / 2.0 * 255.0
        return img.clip(0, 255).astype(np.uint8)

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        task_ids: Sequence[Any],
        seeds: Sequence[int],
    ) -> None:
        for row, (env_id, task_id, seed) in enumerate(zip(env_ids, task_ids, seeds)):
            sid = f"{uuid.uuid4().hex[:12]}-{int(env_id)}"
            self._sessions[int(env_id)] = {
                "session_id": sid,
                "task_id": task_id,
                "seed": int(seed),
                "frames": [self._to_uint8(f) for f in init_frames[row]],
                "actions": init_actions[row].detach().cpu().clone(),
                "step_id": 0,
            }
            self._transport.open(sid, task_id, int(seed))

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            session = self._sessions.pop(int(env_id), None)
            if session is not None:
                self._transport.close(session["session_id"])

    def _window_actions(
        self, env_ids: Sequence[int], actions: torch.Tensor
    ) -> torch.Tensor:
        """Prepend each session's action history to its chunk. Does not mutate the history."""
        history = torch.stack(
            [self._sessions[int(i)]["actions"] for i in env_ids], dim=0
        ).to(dtype=actions.dtype)
        if not self.retain_action:
            return actions
        return torch.cat([history, actions], dim=1)

    def _roll_forward(
        self, env_id: int, windowed: torch.Tensor, frames: np.ndarray
    ) -> None:
        """Advance a session's window and action history. Only called after a success."""
        session = self._sessions[int(env_id)]
        keep = self.condition_frame_length - 1
        session["frames"][1:] = list(frames[-keep:])
        tail = windowed[-keep:, :]
        session["actions"][1 : self.condition_frame_length] = tail.to(
            dtype=session["actions"].dtype
        )
        session["step_id"] += 1

    def generate(
        self,
        env_ids: Sequence[int],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        missing = [int(i) for i in env_ids if int(i) not in self._sessions]
        if missing:
            raise RuntimeError(
                f"generate called for env slots without a session: {missing}"
            )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.detach().cpu().float()
        if actions.shape[0] != len(env_ids):
            raise ValueError(
                f"env_ids and actions must describe the same batch rows; got "
                f"{len(env_ids)}, {actions.shape[0]}"
            )

        windowed = self._window_actions(env_ids, actions)

        # One request per slot: a generation endpoint carries one trajectory's step, and the
        # engine serialises them anyway (max_num_seqs=1). Batching across slots is a server
        # concern, not something the client can fake.
        videos = []
        for row, env_id in enumerate(env_ids):
            session = self._sessions[int(env_id)]
            cond = np.stack(session["frames"])
            frames = self._step_with_retry(session, windowed[row].numpy(), cond)
            # A transport that rounds the window up to the latent grid gets frames back for
            # its padding too, so index from the front of the window rather than its end.
            if frames.shape[0] < self.num_frames:
                raise WorldModelUnavailable(
                    f"served {frames.shape[0]} frames for a window of {self.num_frames}; "
                    "a short clip means the request fell off the latent grid"
                )
            frames = frames[: self.num_frames]
            new_frames = frames[self.condition_frame_length :]
            self._roll_forward(env_id, windowed[row], frames)

            video = new_frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # [T, 3, H, W]
            videos.append(video.transpose(0, 1))  # [3, T, H, W]

        return torch.stack(videos, dim=0).to(self.device)

    def _step_with_retry(
        self, session: dict[str, Any], actions: np.ndarray, cond: np.ndarray
    ) -> np.ndarray:
        """A step, retried in place. Safe because the window has not advanced yet."""
        last: Exception | None = None
        started = time.monotonic()
        attempts = 0
        while attempts <= self.max_retries:
            attempts += 1
            try:
                return self._transport.step(
                    session_id=session["session_id"],
                    step_id=session["step_id"],
                    cond_frames=cond,
                    actions=actions,
                    seed=session["seed"],
                    task=session["task_id"],
                )
            # A server that dies mid-request raises neither URLError nor TimeoutError:
            # `RemoteDisconnected` is a ConnectionResetError, and a truncated body is an
            # HTTPException. Both are retryable, and neither may escape as itself.
            except (OSError, http.client.HTTPException, RuntimeError) as exc:
                last = exc
            # Retries are also bounded by wall clock: attempt count alone lets a hung
            # endpoint hold one env slot for the sum of the per-request timeouts, and
            # generate() walks the slots in turn.
            if time.monotonic() - started >= self.step_budget_s:
                break
        raise WorldModelUnavailable(
            f"step failed, attempts={attempts}, elapsed="
            f"{time.monotonic() - started:.1f}s: {last}"
        ) from last

    def offload(self) -> None:
        """No-op: the weights live in the serving process, not this one."""

    def onload(self) -> None:
        """No-op: see :meth:`offload`."""


class WorldModelUnavailable(RuntimeError):
    """The served model did not answer; the caller should truncate, never hang."""


def _post_multipart(url: str, fields: dict[str, Any], timeout: float) -> bytes:
    """Minimal multipart POST so the backend needs nothing beyond the stdlib."""
    boundary = f"----rlinf{uuid.uuid4().hex}"
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        if isinstance(value, tuple):  # (filename, content_type, raw bytes)
            filename, ctype, raw = value
            buf.write(
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
            )
            buf.write(raw)
        else:
            buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            buf.write(str(value).encode())
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=buf.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _decode_video(raw: bytes) -> np.ndarray:
    """An mp4 payload to ``[T, H, W, 3]`` uint8."""
    import imageio.v3 as iio

    return np.asarray(iio.imread(io.BytesIO(raw), extension=".mp4", plugin="pyav"))


class VideosSyncTransport:
    """Carries a step on a one-shot generation endpoint.

    This is the surface that exists today: ``POST /v1/videos/sync`` with the action riding in
    ``extra_params``. It holds no session, so ``open`` and ``close`` are local no-ops.
    """

    def __init__(self, cfg, remote_cfg):
        self.url = remote_cfg.server_url.rstrip("/") + "/v1/videos/sync"
        self.model = remote_cfg.get("model", "nvidia/Cosmos3-Nano")
        self.timeout = remote_cfg.get("request_timeout_s", 300)
        self.warmup_timeout = remote_cfg.get("warmup_timeout_s", 1800)
        self.domain_name = remote_cfg.get("domain_name", "libero")
        self.num_inference_steps = remote_cfg.get("num_inference_steps", 30)
        self.fps = remote_cfg.get("fps", 5)
        self.prompt = remote_cfg.get("prompt", "A robot arm manipulates objects.")
        self.num_frames = cfg.num_frames
        self.image_size = tuple(cfg.image_size)
        # The first request for an output shape pays a JIT build; only that one gets the long
        # timeout, so a genuinely stuck steady-state step still surfaces as a failure. A
        # server that is not up yet fails at connect, not at read, so it does not spend this.
        self._warm = False

    def open(self, session_id: str, task: Any, seed: int) -> None:
        return None

    def close(self, session_id: str) -> None:
        return None

    def step(
        self,
        session_id: str,
        step_id: int,
        cond_frames: np.ndarray,
        actions: np.ndarray,
        seed: int,
        task: Any,
    ) -> np.ndarray:
        import imageio.v3 as iio

        first = io.BytesIO()
        iio.imwrite(first, cond_frames[0], extension=".jpg")
        h, w = self.image_size
        extra = {
            "action_mode": "forward_dynamics",
            "domain_name": self.domain_name,
            "raw_action_dim": int(actions.shape[-1]),
            "action_chunk_size": int(actions.shape[0]),
            "action": [[float(x) for x in row] for row in actions],
        }
        fields = {
            "model": self.model,
            "prompt": self.prompt,
            "input_reference": ("first_frame.jpg", "image/jpeg", first.getvalue()),
            "size": f"{w}x{h}",
            "num_frames": self.num_frames,
            "fps": self.fps,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": 1.0,
            "flow_shift": 5.0,
            "extra_params": json.dumps(extra),
            "seed": seed,
        }
        timeout = self.timeout if self._warm else self.warmup_timeout
        started = time.monotonic()
        try:
            raw = _post_multipart(self.url, fields, timeout)
        except (OSError, http.client.HTTPException):
            # The long budget is spent by waiting, not by attempting. A request that hung
            # for it does not get it again, or a dead endpoint would hold the slot for that
            # budget once per retry; a connection refused while the server boots keeps it.
            if time.monotonic() - started >= self.timeout:
                self._warm = True
            raise
        self._warm = True
        return _decode_video(raw)


class BwmVideosSyncTransport:
    """Carries a step on the same one-shot endpoint, shaped for BWM.

    BWM conditions on the whole window rather than a single frame, so the condition frames
    ride as an mp4 in ``input_reference`` and the action spans history plus future. The
    pipeline falls back to ``extra_args["action"]`` when ``multi_modal_data`` carries none,
    which is what ``extra_params`` becomes server-side.
    """

    def __init__(self, cfg, remote_cfg):
        self.url = remote_cfg.server_url.rstrip("/") + "/v1/videos/sync"
        self.model = remote_cfg.model
        self.timeout = remote_cfg.get("request_timeout_s", 300)
        self.warmup_timeout = remote_cfg.get("warmup_timeout_s", 1800)
        self.num_inference_steps = remote_cfg.get("num_inference_steps", 50)
        self.fps = remote_cfg.get("fps", 24)
        self.prompt = remote_cfg.get("prompt", "")
        self.image_size = tuple(cfg.image_size)
        self._warm = False

    def open(self, session_id: str, task: Any, seed: int) -> None:
        return None

    def close(self, session_id: str) -> None:
        return None

    def step(
        self,
        session_id: str,
        step_id: int,
        cond_frames: np.ndarray,
        actions: np.ndarray,
        seed: int,
        task: Any,
    ) -> np.ndarray:
        import imageio.v2 as imageio

        # The latent grid is 4n+1 pixel frames. A window off that grid is truncated server-side
        # without an error, so round up and carry the last action across the padding.
        num_frames = 1 + ((actions.shape[0] - 1 + 3) // 4) * 4
        if actions.shape[0] < num_frames:
            pad = np.repeat(actions[-1:], num_frames - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        history = io.BytesIO()
        writer = imageio.get_writer(history, format="mp4", fps=self.fps, quality=8)
        for frame in cond_frames:
            writer.append_data(frame)
        writer.close()

        h, w = self.image_size
        prompt = task if isinstance(task, str) else self.prompt
        if not prompt:
            # An empty multipart field reads as absent server-side, so the 400 that comes
            # back names the prompt rather than the config that failed to set it.
            raise ValueError(
                "no prompt for the served world model: the env identifies episodes by "
                "index, so world_model.remote.prompt has to carry the task wording"
            )
        fields = {
            "model": self.model,
            "prompt": prompt,
            "input_reference": ("history.mp4", "video/mp4", history.getvalue()),
            "size": f"{w}x{h}",
            "num_frames": num_frames,
            "fps": self.fps,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": 1.0,
            "extra_params": json.dumps(
                {"action": [[float(x) for x in row] for row in actions]}
            ),
            "seed": seed,
        }
        timeout = self.timeout if self._warm else self.warmup_timeout
        started = time.monotonic()
        try:
            raw = _post_multipart(self.url, fields, timeout)
        except (OSError, http.client.HTTPException):
            # Same warmup accounting as VideosSyncTransport.
            if time.monotonic() - started >= self.timeout:
                self._warm = True
            raise
        self._warm = True
        return _decode_video(raw)
