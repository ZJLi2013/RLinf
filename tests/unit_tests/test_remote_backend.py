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

import http.client
import importlib.util
import socket
import sys
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

WINDOW = 5
CHUNK = 8
FRAMES = WINDOW + CHUNK
SIZE = 8


def _load_backend_module(monkeypatch):
    """Load the remote backend by path, so it needs no serving deps to import."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "rlinf" / "envs" / "sim" / "world_model" / "remote_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.sim.world_model.remote_backend", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class RecordingTransport:
    """A transport that records what the backend asked it to send.

    The window and the action history are the backend's own state, so what the server would
    have received is the observable side of them.
    """

    def __init__(self, fail_times=0, delay=0.0, error=None):
        self.calls = []
        self.opened = []
        self.closed = []
        self.fail_times = fail_times
        self.delay = delay
        self.error = error or urllib.error.URLError("injected")

    def open(self, session_id, task, seed):
        self.opened.append((session_id, task, seed))

    def close(self, session_id):
        self.closed.append(session_id)

    def step(self, session_id, step_id, cond_frames, actions, seed, task):
        self.calls.append(
            {
                "session_id": session_id,
                "step_id": step_id,
                "cond": cond_frames.copy(),
                "actions": actions.copy(),
                "seed": seed,
            }
        )
        if self.delay:
            time.sleep(self.delay)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        base = 100 + 10 * len(self.calls)
        return np.stack(
            [np.full((SIZE, SIZE, 3), base + i, dtype=np.uint8) for i in range(FRAMES)]
        )


def _make_backend(module, transport, max_retries=2, step_budget_s=900):
    backend = object.__new__(module.RemoteWorldModelBackend)
    backend.device = torch.device("cpu")
    backend.chunk = CHUNK
    backend.condition_frame_length = WINDOW
    backend.num_frames = FRAMES
    backend.image_size = (SIZE, SIZE)
    backend.retain_action = True
    backend.max_retries = max_retries
    backend.step_budget_s = step_budget_s
    backend._transport = transport
    backend._sessions = {}
    return backend


def _open(backend, env_ids=(0,), seeds=(3,)):
    frames = [
        [torch.full((3, 1, SIZE, SIZE), 0.1 * (i + 1)) for i in range(WINDOW)]
        for _ in env_ids
    ]
    backend.open_session(
        env_ids=list(env_ids),
        init_frames=frames,
        init_actions=torch.zeros(len(env_ids), WINDOW, 7),
        task_ids=[7] * len(env_ids),
        seeds=list(seeds),
    )


def _actions(rows=1, seed=2):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(-0.3, 0.3, (rows, CHUNK, 7)).astype(np.float32))


def test_open_session_hands_the_transport_one_session_per_slot(monkeypatch):
    module = _load_backend_module(monkeypatch)
    transport = RecordingTransport()
    backend = _make_backend(module, transport)
    _open(backend, env_ids=[0, 1], seeds=[3, 4])

    assert [seed for _, _, seed in transport.opened] == [3, 4]
    assert len({sid for sid, _, _ in transport.opened}) == 2

    backend.close_session([1])
    assert transport.closed == [transport.opened[1][0]]
    assert set(backend._sessions) == {0}


def test_generate_rejects_a_batch_that_does_not_line_up(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module, RecordingTransport())
    _open(backend, env_ids=[0, 1], seeds=[0, 0])

    with pytest.raises(ValueError):
        backend.generate(env_ids=[0, 1], actions=_actions(rows=1))


def test_generate_needs_a_session(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module, RecordingTransport())

    with pytest.raises(RuntimeError):
        backend.generate(env_ids=[0], actions=_actions())


def test_generate_returns_only_the_new_frames(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module, RecordingTransport())
    _open(backend)

    videos = backend.generate(env_ids=[0], actions=_actions())

    assert videos.shape == (1, 3, CHUNK, SIZE, SIZE)
    assert videos.min() >= -1.0 and videos.max() <= 1.0
    # frame 110 + WINDOW is the first frame past the repainted condition window
    first_new = (110 + WINDOW) / 255.0 * 2.0 - 1.0
    assert videos[0, 0, 0].max().item() == pytest.approx(first_new, abs=1e-6)


def test_a_step_sends_the_window_and_the_action_history(monkeypatch):
    module = _load_backend_module(monkeypatch)
    transport = RecordingTransport()
    backend = _make_backend(module, transport)
    _open(backend)

    backend.generate(env_ids=[0], actions=_actions())
    backend.generate(env_ids=[0], actions=_actions(seed=5))

    first, second = transport.calls
    assert first["step_id"] == 0 and second["step_id"] == 1
    assert first["seed"] == 3
    # retain_action prepends the history, so a step carries window + chunk actions
    assert first["actions"].shape == (WINDOW + CHUNK, 7)
    # the reference frame stays; the rest of the window is the served frames
    assert np.array_equal(first["cond"][0], second["cond"][0])
    assert not np.array_equal(first["cond"][1], second["cond"][1])
    assert not np.array_equal(first["actions"], second["actions"])


def test_a_retried_step_does_not_advance_twice(monkeypatch):
    module = _load_backend_module(monkeypatch)
    transport = RecordingTransport(fail_times=1)
    backend = _make_backend(module, transport)
    _open(backend)

    backend.generate(env_ids=[0], actions=_actions())

    assert len(transport.calls) == 2
    retried, succeeded = transport.calls
    assert retried["step_id"] == succeeded["step_id"] == 0
    assert np.array_equal(retried["cond"], succeeded["cond"])
    assert np.array_equal(retried["actions"], succeeded["actions"])
    assert backend._sessions[0]["step_id"] == 1


def test_exhausted_retries_leave_the_session_where_it_was(monkeypatch):
    module = _load_backend_module(monkeypatch)
    transport = RecordingTransport(fail_times=99)
    backend = _make_backend(module, transport)
    _open(backend)
    before_window = [f.copy() for f in backend._sessions[0]["frames"]]
    before_actions = backend._sessions[0]["actions"].clone()

    with pytest.raises(module.WorldModelUnavailable):
        backend.generate(env_ids=[0], actions=_actions())

    assert len(transport.calls) == 3  # max_retries=2 means three attempts
    assert backend._sessions[0]["step_id"] == 0
    assert all(
        np.array_equal(a, b)
        for a, b in zip(before_window, backend._sessions[0]["frames"])
    )
    assert torch.equal(before_actions, backend._sessions[0]["actions"])


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("refused"),
        TimeoutError("read timed out"),
        http.client.RemoteDisconnected("closed mid-request"),
        http.client.IncompleteRead(b"", 10),
    ],
)
def test_every_shape_of_a_dying_endpoint_is_retryable(monkeypatch, error):
    """A server that closes mid-request raises neither URLError nor TimeoutError."""
    module = _load_backend_module(monkeypatch)
    transport = RecordingTransport(fail_times=1, error=error)
    backend = _make_backend(module, transport)
    _open(backend)

    backend.generate(env_ids=[0], actions=_actions())

    assert len(transport.calls) == 2
    assert backend._sessions[0]["step_id"] == 1


def test_retries_stop_at_the_wall_clock_budget(monkeypatch):
    """Attempt count alone lets a hung endpoint cost the sum of its timeouts."""
    module = _load_backend_module(monkeypatch)
    transport = RecordingTransport(fail_times=99, delay=0.2)
    backend = _make_backend(module, transport, max_retries=10, step_budget_s=0.1)
    _open(backend)

    started = time.monotonic()
    with pytest.raises(module.WorldModelUnavailable):
        backend.generate(env_ids=[0], actions=_actions())

    assert len(transport.calls) == 1
    assert time.monotonic() - started < 1.0


def test_offload_is_a_no_op(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module, RecordingTransport())
    _open(backend)

    backend.offload()
    backend.onload()

    assert set(backend._sessions) == {0}


# --- the wire surface, which owns the timeout budget ------------------------------------


def _dead_url():
    """A port nothing listens on, so a connect fails without waiting."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


class _SilentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hang_s = 1.0

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        time.sleep(self.hang_s)
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _transport(module, url, **remote):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"num_frames": FRAMES, "image_size": [SIZE, SIZE]})
    return module.VideosSyncTransport(
        cfg, OmegaConf.create({"server_url": url, **remote})
    )


def _step(transport):
    transport.step(
        session_id="s",
        step_id=0,
        cond_frames=np.zeros((WINDOW, SIZE, SIZE, 3), dtype=np.uint8),
        actions=np.zeros((FRAMES, 7), dtype=np.float32),
        seed=0,
        task="t",
    )


def test_a_fast_failure_keeps_the_first_request_budget(monkeypatch):
    """A server still booting refuses the connection; it must not spend the build budget."""
    pytest.importorskip("imageio")
    module = _load_backend_module(monkeypatch)
    transport = _transport(
        module, _dead_url(), request_timeout_s=1, warmup_timeout_s=30
    )

    with pytest.raises((OSError, http.client.HTTPException)):
        _step(transport)

    assert transport._warm is False


def test_waiting_out_the_timeout_spends_the_first_request_budget(monkeypatch):
    """Charging the long budget per attempt lets a hung endpoint claim it on every retry."""
    pytest.importorskip("imageio")
    module = _load_backend_module(monkeypatch)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SilentHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        transport = _transport(module, url, request_timeout_s=0.2, warmup_timeout_s=0.4)
        with pytest.raises((OSError, http.client.HTTPException)):
            _step(transport)
        assert transport._warm is True
    finally:
        server.shutdown()
