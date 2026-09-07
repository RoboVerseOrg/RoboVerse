"""The FastTD3 evaluators record what the handler applied, not where the robot ended up."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.types import ObjectState, RobotState, TensorState
from roboverse_learn.rl.fast_td3.trajectory_record import commanded_action, initial_states, recorded_step


def _robot(root, joint_pos, target):
    return RobotState(
        root_state=root,
        body_names=None,
        body_state=None,
        joint_pos=joint_pos,
        joint_vel=torch.zeros_like(joint_pos),
        joint_pos_target=target,
        joint_vel_target=None,
        joint_effort_target=None,
    )


def _object(root, joint_pos):
    return ObjectState(
        root_state=root,
        body_names=None,
        body_state=None,
        joint_pos=joint_pos,
        joint_vel=None if joint_pos is None else torch.zeros_like(joint_pos),
    )


def _states(with_target: bool, root_z: float = 0.5, target=None) -> TensorState:
    root = torch.tensor([[0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6] * 2)
    if with_target and target is None:
        target = torch.tensor([[0.15, 0.25], [0.35, 0.45]])
    return TensorState(
        objects={"drawer": _object(root.clone(), torch.tensor([[0.0], [0.7]])), "cube": _object(root.clone(), None)},
        robots={
            "arm": _robot(root.clone(), torch.tensor([[0.10, 0.20], [0.30, 0.40]]), target if with_target else None)
        },
        cameras={},
        extras={},
    )


def _handler():
    names = {"arm": ["j1", "j2"], "drawer": ["slide"], "cube": []}
    return SimpleNamespace(get_joint_names=lambda name, sort=True: names[name])


def test_records_are_v2_states_keyed_by_the_handler_joint_order_plus_the_reported_targets():
    records = recorded_step(_handler(), _states(True))
    assert len(records) == 2
    state = records[1].state
    assert state["arm"]["dof_pos"] == {"j1": pytest.approx(0.30), "j2": pytest.approx(0.40)}
    assert state["drawer"]["dof_pos"] == {"slide": pytest.approx(0.7)}, "an articulated object gets its joints too"
    assert set(state["cube"]) == {"pos", "rot"} and isinstance(state["cube"]["pos"], list)
    assert "dof_pos_target" not in state["arm"], "a target is a control input, not part of a recorded state"
    assert records[1].targets["arm"] == {"j1": pytest.approx(0.35), "j2": pytest.approx(0.45)}


def test_action_is_the_target_the_handler_applied_and_never_the_achieved_position():
    records = recorded_step(_handler(), _states(True))
    assert commanded_action(records[1], robot_name="arm") == {
        "dof_pos_target": {"j1": pytest.approx(0.35), "j2": pytest.approx(0.45)}
    }
    with pytest.raises(ValueError, match="reports no joint_pos_target"):
        commanded_action(recorded_step(_handler(), _states(False))[0], robot_name="arm")
    with pytest.raises(ValueError, match="'leg' is not in the recorded state"):
        commanded_action(records[0], robot_name="leg")


def test_an_auto_reset_env_is_recorded_from_the_state_its_episode_ended_in():
    """``RLTaskEnv.step`` resets done envs before returning; ``info`` carries their ids and a copy of the
    state before the reset, so the recorder keeps the terminal state instead of the reset pose."""
    after_reset = _states(True, root_z=0.0)
    root = torch.tensor([[0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6])
    terminal = TensorState(
        objects={"drawer": _object(root.clone(), torch.tensor([[0.7]])), "cube": _object(root.clone(), None)},
        robots={"arm": _robot(root.clone(), torch.tensor([[0.99, 0.98]]), torch.tensor([[0.97, 0.96]]))},
        cameras={},
        extras={},
    )
    records = recorded_step(_handler(), after_reset, {"auto_reset_env_ids": [1], "terminal_states": terminal})
    assert records[0].state["arm"]["pos"][2] == 0.0, "env 0 was not reset: the current state"
    assert records[1].state["arm"]["pos"][2] == pytest.approx(0.9)
    assert records[1].state["arm"]["dof_pos"]["j1"] == pytest.approx(0.99)
    assert commanded_action(records[1], robot_name="arm")["dof_pos_target"]["j1"] == pytest.approx(0.97)


def test_an_info_without_the_auto_reset_contract_is_refused():
    with pytest.raises(ValueError, match="carries no 'auto_reset_env_ids'"):
        recorded_step(_handler(), _states(True), {"observations": {}})
    assert len(recorded_step(_handler(), _states(True), {"auto_reset_env_ids": [], "terminal_states": None})) == 2


def test_a_reset_without_the_terminal_copy_is_refused_and_initial_states_read_the_handler():
    with pytest.raises(ValueError, match="record_terminal_states = True"):
        recorded_step(_handler(), _states(True), {"auto_reset_env_ids": [1], "terminal_states": None})
    handler = _handler()
    handler.get_states = lambda mode="tensor": _states(True, root_z=0.3)
    assert [s["arm"]["pos"][2] for s in initial_states(handler)] == [pytest.approx(0.3)] * 2


def test_only_the_requested_envs_are_converted_and_reset_rows_still_land():
    root = torch.tensor([[0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6])
    terminal = TensorState(
        objects={"drawer": _object(root.clone(), torch.tensor([[0.7]])), "cube": _object(root.clone(), None)},
        robots={"arm": _robot(root.clone(), torch.tensor([[0.99, 0.98]]), torch.tensor([[0.97, 0.96]]))},
        cameras={},
        extras={},
    )
    info = {"auto_reset_env_ids": [1], "terminal_states": terminal}
    records = recorded_step(_handler(), _states(True, root_z=0.0), info, env_ids=[1])
    assert len(records) == 1 and records[0].state["arm"]["pos"][2] == pytest.approx(0.9)
    records = recorded_step(_handler(), _states(True, root_z=0.0), info, env_ids=[0])
    assert len(records) == 1 and records[0].state["arm"]["pos"][2] == 0.0, "env 1's reset does not touch env 0"


class _Policy:
    def eval(self):
        pass

    def __call__(self, obs):
        return torch.zeros(obs.shape[0], 2) if obs.ndim == 2 else obs


class _TwoEnvs:
    """Env 0's episodes end every 2 steps and ``step`` auto-resets it the way ``RLTaskEnv.step`` does (its
    row is the reset state, ``info`` carries the terminal copy). Env 1's end every 3 steps but are marked
    done after the base step, as the pick-place release check does, so only an explicit ``reset`` restarts it."""

    num_envs = 2
    max_episode_steps = 3
    robot = SimpleNamespace(name="arm")
    record_terminal_states = False

    def __init__(self):
        self.t = [0, 0]
        self.handler = _handler()
        self.handler.get_states = lambda mode="tensor": self._state
        self._state = self._rows([0, 0])

    @staticmethod
    def _rows(z):
        root = torch.tensor([[0.0, 0.0, float(zi), 1.0, 0.0, 0.0, 0.0] + [0.0] * 6 for zi in z])
        n = len(z)
        return TensorState(
            objects={"drawer": _object(root.clone(), torch.zeros(n, 1)), "cube": _object(root.clone(), None)},
            robots={"arm": _robot(root.clone(), torch.full((n, 2), 0.1), torch.tensor([[float(zi)] * 2 for zi in z]))},
            cameras={},
            extras={},
        )

    def reset(self, env_ids=None):
        for i in range(2) if env_ids is None else env_ids:
            self.t[i] = 0
        self._state = self._rows(self.t)
        return torch.zeros(2, 4), {}

    def step(self, actions):
        self.t = [k + 1 for k in self.t]
        rewards = torch.tensor([float(k) for k in self.t])  # env 0's episodes return 1+2, env 1's 1+2+3
        done_ids = [i for i, period in enumerate((2, 3)) if self.t[i] % period == 0]
        auto_reset = [i for i in done_ids if i == 0]
        terminal = self._rows(self.t)
        for i in auto_reset:
            self.t[i] = 0
        self._state = self._rows(self.t)  # env 0 is back at z=0 when done; env 1 stays where its episode ended
        info = {"auto_reset_env_ids": auto_reset, "terminal_states": None}
        if auto_reset and self.record_terminal_states:
            from metasim.utils.state import select_envs

            info["terminal_states"] = select_envs(terminal, auto_reset)
        terminated = torch.tensor([i in done_ids for i in range(2)])
        return torch.zeros(2, 4), rewards, terminated, torch.zeros(2, dtype=torch.bool), info


def test_evaluate_saves_every_episode_of_every_env_and_none_carries_over(tmp_path):
    """A done env starts its next episode at once (auto-reset inside ``env.step``, or reset by the evaluator
    when the task marked it done after that); the other envs' episodes keep running. Nothing is reset
    globally, so no partial episode is abandoned or spliced."""
    pytest.importorskip("tensordict")  # the evaluator imports fttd3_module, which needs it
    from metasim.utils.demo_util import load_traj_file
    from roboverse_learn.rl.fast_td3.evaluate import evaluate

    stats = evaluate(
        _TwoEnvs(),
        _Policy(),
        _Policy(),
        num_episodes=2,
        device=torch.device("cpu"),
        task_name="fake",
        render=False,
        save_traj=True,
        save_states=True,
        traj_dir=str(tmp_path),
    )
    assert stats["num_episodes"] == 4
    assert stats["mean_return"] == pytest.approx((3 + 3 + 6 + 6) / 4), "a return is one episode's rewards only"
    assert stats["mean_length"] == pytest.approx((2 + 2 + 3 + 3) / 4)
    (path,) = tmp_path.glob("*.pkl")
    episodes = load_traj_file(str(path))["arm"]
    assert [len(ep["actions"]) for ep in episodes] == [2, 2, 3, 3], "env 0's two episodes, then env 1's"
    for ep, last in zip(episodes, (2, 2, 3, 3), strict=True):
        assert len(ep["states"]) == len(ep["actions"])
        assert ep["init_state"]["arm"]["pos"][2] == 0.0, "every episode starts from a reset state"
        assert ep["states"][-1]["arm"]["pos"][2] == pytest.approx(last), "and ends with its terminal state"
        assert ep["actions"][-1]["dof_pos_target"]["j1"] == pytest.approx(last), "with the target of that step"
