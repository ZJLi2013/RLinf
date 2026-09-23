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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import compute_grpo_advantages
from rlinf.data.schema.embodied_types import Trajectory, TrajectoryStep
from rlinf.envs.sim.world_model.backend import WorldModelGeneration
from rlinf.envs.sim.world_model.backend.remote import (
    RemoteCapabilities,
    RemoteStepResult,
    RemoteWorldModelBackend,
)
from rlinf.envs.sim.world_model.env import WorldModelEnv
from rlinf.utils.utils import masked_mean_ratio
from rlinf.workers.actor.embodied_fsdp_actor_worker import (
    EmbodiedFSDPActor,
    _reward_filter_by_group,
)


def _cfg():
    return OmegaConf.create(
        {
            "chunk": 2,
            "condition_frame_length": 2,
            "num_frames": 4,
            "image_size": [2, 3],
            "action_dim": 1,
            "enable_kir": False,
            "retain_action": True,
            "remote": {
                "encoding": "raw",
                "max_retries": 1,
                "step_budget_s": 1,
            },
        }
    )


class FakeTransport:
    def __init__(
        self,
        handler=None,
        *,
        action_dim=1,
        max_batch=2,
        retry_granularity="per_item",
    ):
        self.handler = handler or self._success
        self.calls = []
        self._capabilities = RemoteCapabilities(
            condition_frames=2,
            chunk=2,
            frames_per_step=4,
            image_size=(2, 3),
            action_dim=action_dim,
            output_encodings=("raw",),
            max_batch=max_batch,
            seed_mode="shared_batch",
            retry_granularity=retry_granularity,
            supports_kir=False,
        )

    def capabilities(self):
        return self._capabilities

    def open(self, session_id):
        return None

    def close(self, session_id):
        return None

    def step_batch(self, requests):
        self.calls.append(requests)
        return self.handler(requests, len(self.calls))

    @staticmethod
    def _success(requests, _):
        return [
            RemoteStepResult(
                request_id=request.request_id,
                frames=np.full((4, 2, 3, 3), request.slot_id + 1, dtype=np.uint8),
            )
            for request in requests
        ]


def _open(backend, env_ids=(0, 1)):
    init_frames = [[torch.zeros(3, 1, 2, 3), torch.zeros(3, 1, 2, 3)] for _ in env_ids]
    backend.open_session(
        env_ids=env_ids,
        init_frames=init_frames,
        init_actions=torch.zeros(len(env_ids), 2, 1),
        seeds=[0] * len(env_ids),
    )


def test_capability_mismatch_fails_before_open():
    with pytest.raises(ValueError, match="action_dim=2"):
        RemoteWorldModelBackend(
            _cfg(), torch.device("cpu"), transport=FakeTransport(action_dim=2)
        )


@pytest.mark.parametrize(
    ("server_urls", "expected"),
    [
        (["http://s0"], [0, 0, 0, 0]),
        (["http://s0", "http://s1"], [0, 0, 1, 1]),
        (
            ["http://s0", "http://s1", "http://s2", "http://s3"],
            [0, 1, 2, 3],
        ),
    ],
)
def test_static_server_mapping(server_urls, expected):
    for client_rank, server_index in enumerate(expected):
        cfg = _cfg()
        cfg.remote.server_urls = server_urls
        backend = RemoteWorldModelBackend(
            cfg,
            torch.device("cpu"),
            transport=FakeTransport(),
            client_rank=client_rank,
            num_clients=4,
        )
        assert backend.server_index == server_index
        assert backend.server_url == server_urls[server_index]


def test_generate_uses_one_batch_and_commits_successful_rows():
    transport = FakeTransport()
    backend = RemoteWorldModelBackend(_cfg(), torch.device("cpu"), transport=transport)
    _open(backend)

    result = backend.generate([0, 1], torch.zeros(2, 2, 1))

    assert len(transport.calls) == 1
    assert len(transport.calls[0]) == 2
    assert result.valid.tolist() == [True, True]
    assert result.frames.shape == (2, 3, 2, 2, 3)
    assert backend._sessions[0]["step_id"] == 1
    assert backend._sessions[1]["step_id"] == 1


def test_partial_failure_only_commits_successful_rows():
    def partial(requests, _):
        return [
            RemoteStepResult(
                request_id=requests[0].request_id,
                frames=np.ones((4, 2, 3, 3), dtype=np.uint8),
            ),
            RemoteStepResult(
                request_id=requests[1].request_id,
                error="slot failed",
            ),
        ]

    transport = FakeTransport(partial)
    backend = RemoteWorldModelBackend(_cfg(), torch.device("cpu"), transport=transport)
    _open(backend)

    result = backend.generate([0, 1], torch.zeros(2, 2, 1))

    assert result.valid.tolist() == [True, False]
    assert result.errors == (None, "slot failed")
    assert backend._sessions[0]["step_id"] == 1
    assert backend._sessions[1]["step_id"] == 0


def test_retry_resubmits_only_failed_rows():
    def retry_one(requests, call):
        if call == 1:
            return [
                RemoteStepResult(
                    request_id=requests[0].request_id,
                    frames=np.ones((4, 2, 3, 3), dtype=np.uint8),
                ),
                RemoteStepResult(
                    request_id=requests[1].request_id,
                    error="retry",
                    retryable=True,
                ),
            ]
        return [
            RemoteStepResult(
                request_id=requests[0].request_id,
                frames=np.ones((4, 2, 3, 3), dtype=np.uint8),
            )
        ]

    transport = FakeTransport(retry_one)
    backend = RemoteWorldModelBackend(_cfg(), torch.device("cpu"), transport=transport)
    _open(backend)

    result = backend.generate([0, 1], torch.zeros(2, 2, 1))

    assert result.valid.tolist() == [True, True]
    assert [len(call) for call in transport.calls] == [2, 1]
    assert transport.calls[1][0].slot_id == 1


def test_shared_batch_retry_preserves_membership_and_row_order():
    def retry_one(requests, call):
        if call == 1:
            return [
                RemoteStepResult(
                    request_id=requests[0].request_id,
                    frames=np.ones((4, 2, 3, 3), dtype=np.uint8),
                ),
                RemoteStepResult(
                    request_id=requests[1].request_id,
                    error="retry",
                    retryable=True,
                ),
            ]
        return [
            RemoteStepResult(
                request_id=request.request_id,
                frames=np.ones((4, 2, 3, 3), dtype=np.uint8),
            )
            for request in requests
        ]

    transport = FakeTransport(retry_one, retry_granularity="shared_batch")
    backend = RemoteWorldModelBackend(_cfg(), torch.device("cpu"), transport=transport)
    _open(backend)

    result = backend.generate([0, 1], torch.zeros(2, 2, 1))

    assert result.valid.tolist() == [True, True]
    assert [len(call) for call in transport.calls] == [2, 2]
    assert [request.request_id for request in transport.calls[1]] == [
        request.request_id for request in transport.calls[0]
    ]


class FakeRewardModel:
    def __init__(self):
        self.batch_size = None

    def predict_rew(self, frames):
        self.batch_size = frames.shape[0]
        return torch.ones(frames.shape[0])


class FakeBackend:
    def __init__(self):
        self.result = WorldModelGeneration(
            frames=torch.zeros(2, 3, 2, 2, 3),
            valid=torch.tensor([True, False]),
            errors=(None, "timeout"),
        )

    def generate(self, env_ids, actions):
        return self.result

    def reward_instructions(self, env):
        return None

    def onload(self):
        return None


def test_world_model_failure_truncates_and_masks_slot():
    env = WorldModelEnv.__new__(WorldModelEnv)
    env.backend = FakeBackend()
    env.reward_model = FakeRewardModel()
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.chunk = 2
    env.condition_frame_length = 2
    env.image_size = (2, 3)
    env.current_obs = torch.zeros(2, 3, 1, 2, 2, 3)
    env._elapsed_steps = torch.zeros(2, dtype=torch.long)
    env.prev_step_reward = torch.zeros(2)
    env.returns = torch.zeros(2)
    env.success_once = torch.zeros(2, dtype=torch.bool)
    env.task_descriptions = ["a", "b"]
    env.record_metrics = True
    env.auto_reset = False
    env.use_rel_reward = False
    env._is_offloaded = False
    env.cfg = SimpleNamespace(
        reward_coef=1.0,
        max_episode_steps=10,
        success_reward_threshold=0.9,
    )

    _, rewards, _, truncations, infos = env.chunk_step(torch.zeros(2, 2, 1))

    assert env.reward_model.batch_size == 2
    assert rewards[0].tolist() == [1.0, 1.0]
    assert rewards[1].tolist() == [0.0, 0.0]
    assert truncations[:, -1].tolist() == [False, True]
    assert infos[0]["transition_valid"].tolist() == [
        [True, True],
        [False, False],
    ]
    assert infos[0]["world_model_errors"] == (None, "timeout")

    env.auto_reset = True
    env._handle_auto_reset = lambda dones, obs, infos: (obs, {})
    _, _, _, _, reset_infos = env.chunk_step(torch.zeros(2, 2, 1))
    assert reset_infos[0]["transition_valid"].tolist() == [
        [True, True],
        [False, False],
    ]


def test_invalid_trajectories_do_not_change_grpo_group_statistics():
    rewards = torch.tensor([1.0, 2.0, 100.0, 100.0])
    loss_mask = torch.tensor([[True, True, False, False]])

    advantages, _ = compute_grpo_advantages(rewards, loss_mask, group_size=4)

    assert torch.isfinite(advantages).all()
    assert advantages[0, 0] < 0
    assert advantages[0, 1] > 0
    assert advantages[0, 2:].tolist() == [0.0, 0.0]


def test_validity_survives_main_trajectory_api():
    valids = torch.tensor([[True, False]])
    trajectory = Trajectory.from_steps([TrajectoryStep(valids=valids)])

    assert trajectory.valids.shape == (1, 1, 2)
    assert Trajectory.to_batch([trajectory])["valids"].equal(valids.unsqueeze(0))


def test_reward_filter_ignores_invalid_trajectory_rewards():
    rewards = torch.tensor([[[2.5, 2.5], [2.5, 2.5], [50.0, 50.0], [50.0, 50.0]]])
    loss_mask = torch.tensor([[[True], [True], [False], [False]]])

    reward_filter = _reward_filter_by_group(
        rewards,
        loss_mask,
        group_size=4,
        lower_bound=0.5,
        upper_bound=4.5,
    )

    assert not reward_filter.any()


def _process_rollout(valids, auto_reset):
    actor = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "env": {
                    "train": {
                        "rollout_epoch": 1,
                        "auto_reset": auto_reset,
                        "ignore_terminations": auto_reset,
                    }
                },
                "algorithm": {"reward_type": "action_level", "filter_rewards": False},
            }
        )
    )
    batch = {
        "rewards": torch.zeros(2, 2, 2),
        "dones": torch.zeros(3, 2, 2, dtype=torch.bool),
        "valids": valids,
    }
    return EmbodiedFSDPActor._process_received_rollout_batch(actor, batch)


@pytest.mark.parametrize("auto_reset", [True, False])
def test_all_valid_rollout_keeps_default_masks(auto_reset):
    batch = _process_rollout(torch.ones(2, 2, 2, dtype=torch.bool), auto_reset)
    default = _process_rollout(None, auto_reset)

    assert ("loss_mask" in batch) == ("loss_mask" in default)
    assert ("loss_mask_sum" in batch) == ("loss_mask_sum" in default)
    if "loss_mask" in default:
        assert batch["loss_mask"].equal(default["loss_mask"])
        assert batch["loss_mask_sum"].equal(default["loss_mask_sum"])


def test_invalid_rollout_masks_without_adding_loss_mask_sum():
    valids = torch.tensor(
        [[[True, True], [False, False]], [[True, True], [True, True]]]
    )

    batch = _process_rollout(valids, auto_reset=True)

    assert batch["loss_mask"].equal(valids)
    assert "loss_mask_sum" not in batch


def test_invalid_rollout_updates_done_mask_sum():
    valids = torch.tensor(
        [[[True, True], [False, False]], [[True, True], [True, True]]]
    )

    batch = _process_rollout(valids, auto_reset=False)

    assert batch["loss_mask"].equal(valids)
    assert batch["loss_mask_sum"][0, :, 0].tolist() == [4, 2]


def test_zero_loss_mask_ratio_is_finite():
    values = torch.ones(2, 2)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    ratio = torch.zeros(2, 2)

    loss = masked_mean_ratio(values, mask, ratio)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
