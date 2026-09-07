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

"""Replace Qwen-VL's patch-embedding Conv3d with the equivalent linear projection.

On ROCm 6.4 a `Conv3d` over a `(N, 3, 2, 16, 16)` bf16 input segfaults the process, with
no Python traceback; the vision tower is the only place Qwen3-VL uses one.

The substitution is exact. `Qwen3VLVisionPatchEmbed` reshapes its input to one block per
patch and the conv's kernel_size equals its stride equals the full extent of a block, so
every output element is a single dot product of one flattened block with one flattened
filter -- what `F.linear` computes. Only the reduction order differs.
"""

import torch.nn.functional as F

from rlinf.utils.logging import get_logger

_PATCHED = set()

_TARGETS = (
    ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionPatchEmbed"),
    ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VisionPatchEmbed"),
)


def _linear_forward(self, hidden_states):
    proj = self.proj
    weight = proj.weight.reshape(proj.out_channels, -1)
    x = hidden_states.reshape(-1, weight.shape[1]).to(weight.dtype)
    return F.linear(x, weight, proj.bias)


def patch_vision_patch_embed() -> int:
    """Swap the forward of every Qwen-VL patch-embed class the install exposes."""
    for module_name, class_name in _TARGETS:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        if cls in _PATCHED:
            continue
        cls.forward = _linear_forward
        _PATCHED.add(cls)
        get_logger().info("Patched %s to use a linear patch embedding", class_name)
    return len(_PATCHED)
