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

"""Serve a WoVR diffsynth pipeline through a lossless batch endpoint."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("wovr_server")

CONDITION_FRAMES = 5
CHUNK = 8
FRAMES_PER_STEP = 13
ACTION_DIM = 7
IMAGE_SIZE = (256, 256)


class WoVRModel:
    def __init__(self, checkpoint: str, device: str):
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        self.device = torch.device(device)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=str(self.device),
            model_configs=[
                ModelConfig(
                    path=os.path.join(checkpoint, "model-00001.safetensors"),
                    offload_device="cpu",
                ),
                ModelConfig(
                    path=os.path.join(checkpoint, "Wan2.2_VAE.pth"),
                    offload_device="cpu",
                ),
            ],
        )
        self.pipe.dit.to(self.device)
        self.pipe.vae.to(self.device)
        self._lock = threading.Lock()

    def generate(
        self,
        condition_frames: np.ndarray,
        actions: np.ndarray,
        seed: int,
        inference_steps: int,
    ) -> np.ndarray:
        input_images = []
        input_images4 = []
        for window in condition_frames:
            images = [Image.fromarray(frame) for frame in window]
            input_images.append(images[0])
            input_images4.append(images[-4:])

        with self._lock:
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output = self.pipe(
                    seed=seed,
                    tiled=False,
                    input_image=input_images,
                    input_image4=input_images4,
                    action=torch.from_numpy(actions).to(self.device),
                    height=IMAGE_SIZE[0],
                    width=IMAGE_SIZE[1],
                    num_frames=FRAMES_PER_STEP,
                    num_inference_steps=inference_steps,
                    cfg_scale=1.0,
                    progress_bar_cmd=lambda value: value,
                    batch_size=len(condition_frames),
                )
        return np.stack(
            [
                np.stack([np.asarray(frame, dtype=np.uint8) for frame in trajectory])
                for trajectory in output
            ]
        )


def _encode_npz(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


class WoVRHandler(BaseHTTPRequestHandler):
    model: WoVRModel
    model_id: str
    max_batch: int

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, body: dict) -> None:
        self._send(status, json.dumps(body).encode(), "application/json")

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ready": True})
            return
        if self.path == f"/v1/models/{self.model_id}/capabilities":
            self._send_json(
                200,
                {
                    "condition_frames": CONDITION_FRAMES,
                    "chunk": CHUNK,
                    "frames_per_step": FRAMES_PER_STEP,
                    "image_size": list(IMAGE_SIZE),
                    "action_dim": ACTION_DIM,
                    "output_encodings": ["raw"],
                    "max_batch": self.max_batch,
                    "seed_mode": "shared_batch",
                    "retry_granularity": "shared_batch",
                    "supports_kir": False,
                },
            )
            return
        self._send_json(404, {"error": self.path})

    def do_POST(self):
        if self.path != "/v1/world-model/batch":
            self._send_json(404, {"error": self.path})
            return
        try:
            response = self._generate_batch()
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except Exception as error:
            logger.exception("world-model batch failed")
            self._send_json(500, {"error": f"{type(error).__name__}: {error}"})
            return
        self._send(200, response, "application/octet-stream")

    def _generate_batch(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        with np.load(
            io.BytesIO(self.rfile.read(length)), allow_pickle=False
        ) as request:
            request_ids = request["request_ids"].astype(str)
            condition_frames = request["condition_frames"]
            actions = request["actions"].astype(np.float32)
            seeds = request["seeds"].astype(np.int64)
            inference_steps = int(request["inference_steps"].item())

        batch_size = len(request_ids)
        if not 1 <= batch_size <= self.max_batch:
            raise ValueError(
                f"batch size {batch_size} is outside [1, {self.max_batch}]"
            )
        if condition_frames.shape != (
            batch_size,
            CONDITION_FRAMES,
            *IMAGE_SIZE,
            3,
        ):
            raise ValueError(f"unexpected condition shape {condition_frames.shape}")
        if actions.shape != (batch_size, FRAMES_PER_STEP, ACTION_DIM):
            raise ValueError(f"unexpected action shape {actions.shape}")
        if len(set(seeds.tolist())) != 1:
            raise ValueError("WoVR batches require one shared seed")

        frames = self.model.generate(
            condition_frames,
            actions,
            int(seeds[0]),
            inference_steps,
        )
        return _encode_npz(request_ids=request_ids, frames=frames)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="wovr-libero-spatial")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--max-batch", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    WoVRHandler.model = WoVRModel(args.checkpoint, args.device)
    WoVRHandler.model_id = args.model_id
    WoVRHandler.max_batch = args.max_batch
    server = ThreadingHTTPServer((args.host, args.port), WoVRHandler)
    logger.info("serving %s on %s:%d", args.model_id, args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
