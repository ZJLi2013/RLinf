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

"""A frozen VLM as a world-model reward, read off one token's log-probability.

Progress is ``log P(" True")`` after a prompt that shows the frames and then *asserts*
the task was completed, following TOPReward. The assertion framing is not
interchangeable with a question: an invented "is the task complete:" prompt scored a
nonsense instruction higher than the correct one on the same frames.

``predict_rew`` returns a 0/1 label rather than the probability, so the label means the
same thing as the ResNet reward model's ``round()``: it drives ``terminations`` and the
loss mask that truncates the episode there.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes the "
    "following task: "
)
PROMPT_SUFFIX = (
    "{instruction} Decide whether the above statement is True or not. The answer is:"
)
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _import_layered_transformers(lib_dir: Optional[str]):
    """Import a transformers that lives outside the interpreter's site-packages.

    Qwen3-VL needs transformers >= 4.57 while OpenVLA-OFT only loads under the 4.40
    line, and both are reachable from the same launcher, so the newer libs cannot go on
    a cluster-wide PYTHONPATH. Env workers are their own processes and the Wan pipeline
    binds every transformers class it uses at import time, so shadowing the module here
    leaves those bindings intact while everything imported afterwards -- including the
    lazy submodule loads the auto classes do -- resolves against the newer version.
    """
    if not lib_dir:
        import transformers

        return transformers

    for name in [
        name
        for name in sys.modules
        if name.split(".")[0] in ("transformers", "tokenizers")
    ]:
        del sys.modules[name]
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    import transformers

    return transformers


class TOPRewardVLM(nn.Module):
    """Score generated frames with a frozen VLM, no training and no generation."""

    def __init__(
        self,
        model_path: str,
        success_prob_threshold: float = 0.46,
        window_frames: int = 16,
        max_frames: int = 16,
        fps: float = 2.0,
        chunk: int = 8,
        lib_dir: Optional[str] = None,
        attn_implementation: str = "eager",
        dtype: str = "bfloat16",
    ):
        super().__init__()
        self.threshold = float(success_prob_threshold)
        self.window_frames = int(window_frames)
        self.max_frames = int(max_frames)
        self.fps = float(fps)
        self.chunk = int(chunk)
        self._history: dict[int, np.ndarray] = {}

        transformers = _import_layered_transformers(
            lib_dir or os.environ.get("RLINF_VLM_LIB_DIR")
        )
        from rlinf.models.embodiment.reward.qwen3vl_rocm_patch import (
            patch_vision_patch_embed,
        )

        patch_vision_patch_embed()

        auto_vlm = getattr(
            transformers, "AutoModelForImageTextToText", None
        ) or getattr(transformers, "AutoModelForVision2Seq")
        self.processor = transformers.AutoProcessor.from_pretrained(model_path)
        self.model = auto_vlm.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
        )
        self.model.requires_grad_(False)

        true_ids = self.processor.tokenizer.encode(" True", add_special_tokens=False)
        if len(true_ids) != 1:
            raise ValueError(
                f"' True' tokenizes to {len(true_ids)} tokens for {model_path}; "
                "the score reads a single next token"
            )
        self.true_id = true_ids[0]

    def reset_history(self, env_idx=None) -> None:
        """Drop the cached frames of restarted slots, so a window never spans episodes."""
        if env_idx is None:
            self._history.clear()
            return
        for slot in env_idx:
            self._history.pop(int(slot), None)

    def _to_pil(self, frames_chw: np.ndarray):
        from PIL import Image

        if len(frames_chw) > self.max_frames:
            keep = np.linspace(0, len(frames_chw) - 1, self.max_frames).astype(int)
            frames_chw = frames_chw[keep]
        return [
            Image.fromarray(np.transpose(f, (1, 2, 0))[..., :3]).convert("RGB")
            for f in frames_chw
        ]

    @torch.no_grad()
    def _score(self, frames_chw: np.ndarray, instruction: str) -> float:
        text = VIDEO_PLACEHOLDER + PROMPT_PREFIX + PROMPT_SUFFIX.format(
            instruction=instruction
        )
        inputs = self.processor(
            text=[text],
            videos=[self._to_pil(frames_chw)],
            fps=self.fps,
            return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)[self.true_id].item()

    @torch.no_grad()
    def predict_rew(self, images: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        """Label every frame of a chunk with whether that chunk looks complete.

        ``images`` is ``[num_envs * chunk, 3, H, W]`` in ``[-1, 1]`` and ``instructions``
        holds one task description per frame. One forward covers a whole chunk, with the
        previous chunk prepended so the call sees how the episode got here, and its label
        is broadcast back over the chunk's frames.
        """
        num_frames = images.shape[0]
        num_envs = num_frames // self.chunk
        frames = (
            (images.detach().float().clamp(-1.0, 1.0) + 1.0)
            .mul_(127.5)
            .to(torch.uint8)
            .cpu()
            .numpy()
            .reshape(num_envs, self.chunk, *images.shape[1:])
        )

        labels = torch.zeros(num_envs, dtype=images.dtype, device=images.device)
        for env in range(num_envs):
            chunk = frames[env]
            past = self._history.get(env)
            window = chunk if past is None else np.concatenate([past, chunk])
            keep = max(1, self.window_frames - self.chunk)
            self._history[env] = chunk[-keep:]

            logp = self._score(window[-self.window_frames :], instructions[env * self.chunk])
            labels[env] = float(np.exp(logp) >= self.threshold)

        return labels.repeat_interleave(self.chunk)
