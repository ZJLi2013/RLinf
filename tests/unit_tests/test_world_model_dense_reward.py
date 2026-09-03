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

"""A world-model episode returns ``reward_coef * max(p)`` when scored densely.

Success in a world model is not an absorbing state: the reward model's probability peaks
and falls back once generation drifts out of the success pose. A telescoping reward built
on the last frame therefore cancels to zero, which is what these tests pin down.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_ENVS = 4
CHUNK = 8
COEF = 5.0


def _load_env_module(monkeypatch):
    """Load the shared env module; only the dataset wrapper needs stubbing out."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "rlinf" / "envs" / "world_model" / "world_model_env.py"

    fake_dataset = types.ModuleType("rlinf.data.datasets.world_model")
    fake_dataset.NpyTrajectoryDatasetWrapper = object
    monkeypatch.setitem(sys.modules, "rlinf.data.datasets.world_model", fake_dataset)

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.world_model.world_model_env", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_env(module, dense_reward):
    """Only the scoring attributes; _calc_step_reward touches nothing else."""

    class _Env(module.WorldModelEnv):
        def _build_backend(self):
            raise AssertionError("these tests never generate frames")

        def _load_reward_model(self):
            raise AssertionError("these tests feed the probability in directly")

    env = object.__new__(_Env)
    env.cfg = types.SimpleNamespace(reward_coef=COEF)
    env.device = torch.device("cpu")
    env.num_envs = NUM_ENVS
    env.chunk = CHUNK
    env.use_rel_reward = True
    env.dense_reward = dense_reward
    env._chunk_probs = None
    env.prev_step_reward = torch.zeros(NUM_ENVS)
    return env


def _score_episode(env, prob_chunks, rounded_chunks):
    """Drive the per-chunk scoring the way chunk_step does, chunk by chunk."""
    diffs = []
    for probs, rounded in zip(prob_chunks, rounded_chunks):
        env._chunk_probs = probs
        diffs.append(env._calc_step_reward(rounded))
    return torch.stack(diffs)


def _rise_then_fall():
    """One episode per env: the probability peaks in the middle chunk, then decays.

    Peaks differ across the four slots so a group would still see spread even when no
    slot ever crosses the success threshold.
    """
    peaks = torch.tensor([0.9010, 0.2910, 0.1020, 0.0036])
    ramps = torch.tensor([0.02, 0.35, 1.0, 0.6, 0.2, 0.001])
    prob_chunks = [
        (peaks.unsqueeze(1) * ramp).expand(NUM_ENVS, CHUNK).clone().contiguous()
        for ramp in ramps
    ]
    rounded_chunks = [(chunk >= 0.9).float() for chunk in prob_chunks]
    return prob_chunks, rounded_chunks, peaks


@pytest.fixture
def module(monkeypatch):
    return _load_env_module(monkeypatch)


def test_dense_return_is_coef_times_max_prob(module):
    env = _make_env(module, dense_reward=True)
    prob_chunks, rounded_chunks, peaks = _rise_then_fall()

    diffs = _score_episode(env, prob_chunks, rounded_chunks)

    expected = COEF * torch.stack(prob_chunks).amax(dim=(0, 2))
    assert torch.allclose(diffs.sum(dim=(0, 2)), expected, atol=1e-5)
    assert torch.allclose(expected, COEF * peaks, atol=1e-5)


def test_dense_scores_the_probability_not_the_rounded_label(module):
    """The rounded label is all zeros here, so a return of zero means probs were ignored."""
    env = _make_env(module, dense_reward=True)
    prob_chunks, _, _ = _rise_then_fall()
    all_failed = [torch.zeros(NUM_ENVS, CHUNK) for _ in prob_chunks]

    diffs = _score_episode(env, prob_chunks, all_failed)

    assert (diffs.sum(dim=(0, 2)) > 0).all()


def test_dense_reward_is_monotone_so_a_drift_back_cannot_cancel_it(module):
    env = _make_env(module, dense_reward=True)
    prob_chunks, rounded_chunks, _ = _rise_then_fall()

    diffs = _score_episode(env, prob_chunks, rounded_chunks)

    assert (diffs >= 0).all()


def test_binary_reward_cancels_when_success_is_revoked(module):
    """Why max-p is needed: the telescoped 0/1 reward sums to zero once success falls back."""
    env = _make_env(module, dense_reward=False)
    prob_chunks, rounded_chunks, _ = _rise_then_fall()

    diffs = _score_episode(env, prob_chunks, rounded_chunks)

    assert torch.allclose(diffs.sum(dim=(0, 2)), torch.zeros(NUM_ENVS), atol=1e-6)


def test_binary_reward_telescopes_to_the_last_chunk(module):
    env = _make_env(module, dense_reward=False)
    latched = [torch.zeros(NUM_ENVS, CHUNK), torch.ones(NUM_ENVS, CHUNK)]

    diffs = _score_episode(env, latched, latched)

    assert torch.allclose(diffs.sum(dim=(0, 2)), torch.full((NUM_ENVS,), COEF))
