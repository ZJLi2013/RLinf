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

"""Backend that advances frames on a WoVR world model served over HTTP.

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
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import torch

from rlinf.envs.sim.world_model.registry import register_backend

from . import FrameQueue, WorldModelGeneration

__all__ = [
    "RemoteCapabilities",
    "RemoteStepRequest",
    "RemoteStepResult",
    "RemoteWorldModelBackend",
]


@dataclass(frozen=True)
class RemoteCapabilities:
    condition_frames: int
    chunk: int
    frames_per_step: int
    image_size: tuple[int, int]
    action_dim: int
    output_encodings: tuple[str, ...]
    max_batch: int
    seed_mode: str
    retry_granularity: str
    supports_kir: bool


@dataclass(frozen=True)
class RemoteStepRequest:
    request_id: str
    slot_id: int
    session_id: str
    step_id: int
    cond_frames: np.ndarray
    actions: np.ndarray
    seed: int


@dataclass(frozen=True)
class RemoteStepResult:
    request_id: str
    frames: Optional[np.ndarray] = None
    error: Optional[str] = None
    retryable: bool = False


class WorldModelTransport(Protocol):
    """Carries backend batches over a wire surface."""

    def capabilities(self) -> RemoteCapabilities: ...

    def open(self, session_id: str) -> None: ...

    def step_batch(
        self, requests: Sequence[RemoteStepRequest]
    ) -> Sequence[RemoteStepResult]:
        """Return one result per request."""

    def close(self, session_id: str) -> None: ...


@register_backend("remote")
class RemoteWorldModelBackend:
    """Advances frames on a served world model.

    Args:
        cfg: The env config. Generation geometry comes from the usual keys; everything about
            the served model lives under ``world_model.remote``.
        device: Where returned frames land.
    """

    # The served model conditions on the frames it is sent, not on KIR keyframes.
    supports_kir = False

    def __init__(
        self,
        cfg,
        device: torch.device,
        transport: Optional[WorldModelTransport] = None,
        client_rank: int = 0,
        num_clients: int = 1,
    ):
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

        self.remote_cfg = cfg.remote
        self.max_retries = self.remote_cfg.get("max_retries", 2)
        self.step_budget_s = self.remote_cfg.get("step_budget_s", 900)
        self.server_index, self.server_url = self._select_server(
            self.remote_cfg, client_rank, num_clients
        )
        if transport is None:
            name = self.remote_cfg.get("transport", "wovr_batch")
            if name != "wovr_batch":
                raise ValueError(f"unknown transport {name!r}; expected 'wovr_batch'")
            transport_cfg = {key: self.remote_cfg[key] for key in self.remote_cfg}
            transport_cfg["server_url"] = self.server_url
            transport = WoVRBatchTransport(self.cfg, transport_cfg)
        self._transport = transport
        self.capabilities = self._transport.capabilities()
        self._validate_capabilities()
        self._sessions: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _select_server(remote_cfg, client_rank: int, num_clients: int):
        server_urls = remote_cfg.get("server_urls")
        if not server_urls:
            return 0, remote_cfg.get("server_url")
        if num_clients < 1 or not 0 <= client_rank < num_clients:
            raise ValueError(
                f"invalid client rank {client_rank} for {num_clients} clients"
            )
        server_index = client_rank * len(server_urls) // num_clients
        return server_index, str(server_urls[server_index])

    def _validate_capabilities(self) -> None:
        expected = {
            "condition_frames": self.condition_frame_length,
            "chunk": self.chunk,
            "frames_per_step": self.num_frames,
            "image_size": self.image_size,
            "action_dim": self.cfg.get("action_dim", 7),
        }
        mismatches = [
            f"{key}={getattr(self.capabilities, key)!r}, expected {value!r}"
            for key, value in expected.items()
            if getattr(self.capabilities, key) != value
        ]
        encoding = self.remote_cfg.get("encoding", "raw")
        if encoding not in self.capabilities.output_encodings:
            mismatches.append(
                f"encoding={encoding!r}, supported "
                f"{self.capabilities.output_encodings!r}"
            )
        if self.cfg.get("enable_kir", False) and not self.capabilities.supports_kir:
            mismatches.append("enable_kir=True, but the service does not support KIR")
        if self.capabilities.max_batch < 1:
            mismatches.append(
                f"max_batch={self.capabilities.max_batch!r}, expected at least 1"
            )
        if self.capabilities.retry_granularity not in {"per_item", "shared_batch"}:
            mismatches.append(
                f"retry_granularity={self.capabilities.retry_granularity!r}, "
                "expected 'per_item' or 'shared_batch'"
            )
        if mismatches:
            raise ValueError(
                "remote world-model capabilities are incompatible: "
                + "; ".join(mismatches)
            )

    @staticmethod
    def load_reward_model(cfg):
        """The scorer runs here, not on the server, so it is the benchmark's own.

        Imported inside the call: selecting this backend should not require the
        generation stack that happens to ship the reward models.
        """
        from diffsynth.models.reward_model import (
            ResnetRewModel,
            TaskEmbedResnetRewModel,
        )

        if cfg.reward_model.type == "ResnetRewModel":
            return ResnetRewModel(cfg.reward_model.from_pretrained)
        if cfg.reward_model.type == "TaskEmbedResnetRewModel":
            return TaskEmbedResnetRewModel(
                checkpoint_path=cfg.reward_model.from_pretrained,
                task_suite_name=cfg.task_suite_name,
            )
        raise ValueError(f"Unknown reward model type: {cfg.reward_model.type}")

    def reward_instructions(self, env) -> Optional[list[str]]:
        if env.cfg.reward_model.type != "TaskEmbedResnetRewModel":
            return None
        # One instruction per scored frame, so each description repeats over its chunk
        instructions = []
        for env_idx in range(env.num_envs):
            instructions.extend([env.task_descriptions[env_idx]] * self.chunk)
        return instructions

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
        seeds: Sequence[int],
    ) -> None:
        batch_size = len(env_ids)
        if (
            len(init_frames) != batch_size
            or init_actions.shape[0] != batch_size
            or len(seeds) != batch_size
        ):
            raise ValueError(
                "env_ids, init_frames, init_actions and seeds must have the same "
                f"batch size; got {batch_size}, {len(init_frames)}, "
                f"{init_actions.shape[0]}, {len(seeds)}"
            )
        for row, (env_id, seed) in enumerate(zip(env_ids, seeds)):
            sid = f"{uuid.uuid4().hex[:12]}-{int(env_id)}"
            self._sessions[int(env_id)] = {
                "session_id": sid,
                "seed": int(seed),
                "frames": [self._to_uint8(f) for f in init_frames[row]],
                "actions": init_actions[row].detach().cpu().clone(),
                "step_id": 0,
            }
            self._transport.open(sid)

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
    ) -> WorldModelGeneration:
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

        requests = []
        for row, env_id in enumerate(env_ids):
            session = self._sessions[int(env_id)]
            requests.append(
                RemoteStepRequest(
                    request_id=f"{session['session_id']}:{session['step_id']}",
                    slot_id=int(env_id),
                    session_id=session["session_id"],
                    step_id=session["step_id"],
                    cond_frames=np.stack(session["frames"]),
                    actions=windowed[row].numpy(),
                    seed=session["seed"],
                )
            )

        results = []
        max_batch = self.capabilities.max_batch
        for start in range(0, len(requests), max_batch):
            results.extend(
                self._step_batch_with_retry(requests[start : start + max_batch])
            )

        videos = []
        valid = []
        errors = []
        successful = []
        for row, (env_id, result) in enumerate(zip(env_ids, results)):
            frames = result.frames
            error = result.error
            if frames is not None:
                error = self._validate_frames(int(env_id), frames)
            if frames is None or error is not None:
                last_frame = np.asarray(self._sessions[int(env_id)]["frames"][-1])
                new_frames = np.repeat(last_frame[None], self.chunk, axis=0)
                valid.append(False)
                errors.append(error or "remote world-model step failed")
            else:
                new_frames = frames[-self.chunk :]
                valid.append(True)
                errors.append(None)
                successful.append((int(env_id), windowed[row], frames))
            video = new_frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            videos.append(video.transpose(0, 1))

        for env_id, row_actions, frames in successful:
            self._roll_forward(env_id, row_actions, frames)

        frames = torch.stack(videos, dim=0).to(self.device)
        return WorldModelGeneration(
            frames=frames,
            valid=torch.tensor(valid, dtype=torch.bool, device=self.device),
            errors=tuple(errors),
        )

    def _validate_frames(self, env_id: int, frames: np.ndarray) -> Optional[str]:
        h, w = self.image_size
        if frames.ndim != 4:
            return f"served {frames.shape} for env {env_id}, expected [T, H, W, 3]"
        if frames.shape[0] < self.chunk or frames.shape[1:] != (h, w, 3):
            return (
                f"served {frames.shape} for env {env_id}, expected at least "
                f"[{self.chunk}, {h}, {w}, 3]"
            )
        return None

    def _step_batch_with_retry(
        self, requests: Sequence[RemoteStepRequest]
    ) -> list[RemoteStepResult]:
        pending = {request.request_id: request for request in requests}
        request_batch = tuple(requests)
        shared_batch = self.capabilities.retry_granularity == "shared_batch"
        completed: dict[str, RemoteStepResult] = {}
        started = time.monotonic()
        attempts = 0

        while pending and attempts <= self.max_retries:
            attempts += 1
            try:
                batch_results = self._transport.step_batch(
                    request_batch if shared_batch else tuple(pending.values())
                )
            except (OSError, http.client.HTTPException, RuntimeError) as exc:
                batch_results = [
                    RemoteStepResult(
                        request_id=request_id,
                        error=str(exc),
                        retryable=True,
                    )
                    for request_id in pending
                ]

            by_id = {result.request_id: result for result in batch_results}
            retry = {}
            for request_id, request in pending.items():
                result = by_id.get(
                    request_id,
                    RemoteStepResult(
                        request_id=request_id,
                        error="transport omitted this request",
                        retryable=True,
                    ),
                )
                if (
                    result.frames is None
                    and result.retryable
                    and attempts <= self.max_retries
                    and time.monotonic() - started < self.step_budget_s
                ):
                    retry[request_id] = request
                else:
                    completed[request_id] = result
            pending = retry

        elapsed = time.monotonic() - started
        for request_id in pending:
            completed[request_id] = RemoteStepResult(
                request_id=request_id,
                error=f"step failed, attempts={attempts}, elapsed={elapsed:.1f}s",
            )
        return [completed[request.request_id] for request in requests]

    def offload(self) -> None:
        """No-op: the weights live in the serving process, not this one."""

    def onload(self) -> None:
        """No-op: see :meth:`offload`."""


class WoVRBatchTransport:
    """Lossless batch transport for the RLinf WoVR serving endpoint."""

    def __init__(self, cfg, remote_cfg):
        server_url = remote_cfg.get("server_url")
        if not server_url:
            raise ValueError(
                "world_model.remote.server_url is required when world_model.backend=remote"
            )
        self.base_url = server_url.rstrip("/")
        self.url = self.base_url + "/v1/world-model/batch"
        self.model = remote_cfg.get("model", "wovr-libero-spatial")
        self.timeout = remote_cfg.get("request_timeout_s", 300)
        self.inference_steps = remote_cfg.get("num_inference_steps", 5)
        self._capabilities = self._fetch_capabilities()

    def _fetch_capabilities(self) -> RemoteCapabilities:
        url = f"{self.base_url}/v1/models/{self.model}/capabilities"
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            data = json.loads(response.read())
        return RemoteCapabilities(
            condition_frames=int(data["condition_frames"]),
            chunk=int(data["chunk"]),
            frames_per_step=int(data["frames_per_step"]),
            image_size=tuple(data["image_size"]),
            action_dim=int(data["action_dim"]),
            output_encodings=tuple(data["output_encodings"]),
            max_batch=int(data["max_batch"]),
            seed_mode=str(data["seed_mode"]),
            retry_granularity=str(data["retry_granularity"]),
            supports_kir=bool(data["supports_kir"]),
        )

    def capabilities(self) -> RemoteCapabilities:
        return self._capabilities

    def open(self, session_id: str) -> None:
        return None

    def close(self, session_id: str) -> None:
        return None

    def step_batch(
        self, requests: Sequence[RemoteStepRequest]
    ) -> Sequence[RemoteStepResult]:
        seeds = {request.seed for request in requests}
        if len(seeds) != 1:
            raise ValueError(
                f"wovr_batch requires one shared seed, got {sorted(seeds)}"
            )
        buffer = io.BytesIO()
        np.savez(
            buffer,
            request_ids=np.asarray(
                [request.request_id for request in requests], dtype=np.str_
            ),
            condition_frames=np.stack([request.cond_frames for request in requests]),
            actions=np.stack([request.actions for request in requests]),
            seeds=np.asarray([request.seed for request in requests], dtype=np.int64),
            inference_steps=np.asarray(self.inference_steps, dtype=np.int64),
        )
        request = urllib.request.Request(
            self.url,
            data=buffer.getvalue(),
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "application/octet-stream",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        with np.load(io.BytesIO(raw), allow_pickle=False) as result:
            request_ids = result["request_ids"].astype(str).tolist()
            frames = result["frames"]
        by_id = dict(zip(request_ids, frames))
        missing = [item.request_id for item in requests if item.request_id not in by_id]
        if missing:
            raise RuntimeError(f"wovr_batch omitted request ids: {missing}")
        return [
            RemoteStepResult(
                request_id=item.request_id,
                frames=by_id[item.request_id],
            )
            for item in requests
        ]
