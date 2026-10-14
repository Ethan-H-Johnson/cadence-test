"""Behavioral contracts for composing the recovered mechanisms."""

import numpy as np
import pytest

from cadence import Brain, SynapticMemory, Trace


def assert_same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


@pytest.mark.parametrize("observers", [(), (3,), (3, 2)])
def test_modular_base_and_optional_observers_share_one_connected_brain(observers):
    brain = Brain.compose(2, 2, modules=(5, 4), observers=observers, seed=3)
    assert isinstance(brain.working_memory, Trace)
    assert isinstance(brain.hippocampus, SynapticMemory)
    graph = brain.connectome
    neighbors = [set() for _ in range(graph.n)]
    edges = set(zip(graph.pre.tolist(), graph.post.tolist(), strict=True))
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    reached, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(neighbors[node] - reached)
    assert len(reached) == graph.n
    # Ordinary modular depth is present without observation. It is not all-to-all.
    assert not any(
        (a, b) in edges for a in graph.populations["module_0"] for b in graph.populations["motor"]
    )
    for index in range(len(observers)):
        members = graph.populations[f"observer_{index}"]
        assert all(
            (a, b) in edges and (b, a) in edges
            for a in members
            for b in graph.populations["association"]
        )
    phase = brain.imagine([[[0.2, -0.1]]])[0]
    assert np.all(phase.qualified)
    drive = brain.stimulus([[0.2, -0.1]])
    np.testing.assert_allclose(phase.residual, brain.brain.residual(drive, phase.state))


def test_observer_feedback_changes_the_base_inside_the_same_solve():
    brain = Brain.compose(2, 2, modules=(5, 4), observers=(3,), seed=3)
    drive = brain.stimulus([[0.2, -0.1]])
    ordinary = brain.brain.equilibrate(drive, budget=2048, tolerance=1e-8)
    perturbed = drive.copy()
    perturbed[:, brain.connectome.populations["observer_0"]] += 0.1
    feedback = brain.brain.equilibrate(perturbed, budget=2048, tolerance=1e-8)
    assert np.all(ordinary.qualified) and np.all(feedback.qualified)
    base = brain.connectome.populations["association"]
    assert (
        np.max(np.abs(ordinary.state.activation[:, base] - feedback.state.activation[:, base]))
        > 1e-4
    )


def test_imagination_preserves_memories_randomness_pending_outcomes_and_continuation(tmp_path):
    brain = Brain.compose(2, 2, modules=(5, 4), observers=(3,), seed=3)
    brain.hippocampus.observe(np.array([[1.0, 0.0]]), np.array([[0.7, -0.2]]))
    brain.act([[0.2, -0.1]])  # A real action is still awaiting its outcome.
    before = brain.save(tmp_path / "before")
    control = Brain.load(before)
    sequence = [[[0.0, 0.1]], [[-0.1, 0.0]]]
    branch = brain.imagine(sequence)
    assert len(branch) == 2 and all(np.all(phase.qualified) for phase in branch)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    repeated = brain.imagine(sequence)
    for first, second in zip(branch, repeated, strict=True):
        np.testing.assert_array_equal(first.state.activation, second.state.activation)
    # Real feedback still belongs to the preceding real action, with identical learning.
    reward, done, following = np.array([0.4]), np.array([False]), [[0.1, 0.2]]
    brain.learn(reward, done, following)
    control.learn(reward, done, following)
    np.testing.assert_array_equal(brain.act(following), control.act(following))
    assert_same_checkpoint(brain.save(tmp_path / "continued"), control.save(tmp_path / "control"))


def test_private_trace_is_used_between_imagined_steps_without_becoming_real_memory():
    brain = Brain.compose(2, 2, modules=(5, 4), seed=3)
    sequence = [[[0.6, -0.3]], [[0.0, 0.0]]]
    branch = brain.imagine(sequence)
    assert len(branch) == 2 and all(np.all(phase.qualified) for phase in branch)
    assert brain.working_memory.trace.shape[0] == 0
    # Zeroing the trace's read gain removes its causal contribution to the second solve.
    brain.working_memory.amplitude = 0
    erased = brain.imagine(sequence)
    np.testing.assert_array_equal(branch[0].state.activation, erased[0].state.activation)
    assert np.max(np.abs(branch[1].state.activation - erased[1].state.activation)) > 1e-3


def test_refused_imagined_step_ends_branch_and_leaves_real_state_untouched(tmp_path):
    brain = Brain.compose(2, 2, modules=(5, 4), seed=3)
    before = brain.save(tmp_path / "before")
    branch = brain.imagine([[[0.2, 0.1]], [[0.5, 0.3]]], budget=0)
    assert len(branch) == 1 and not np.all(branch[0].qualified)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    with pytest.raises(ValueError, match="stream identities"):
        brain.imagine([[[0.1, 0.2]], [[0.1, 0.2], [0.2, 0.1]]])
    assert_same_checkpoint(before, brain.save(tmp_path / "invalid"))


@pytest.mark.parametrize(
    "modules,observers", [((), ()), ((0,), ()), ((True,), ()), ((4,), (-1,)), (4, ()), ((4,), "2")]
)
def test_invalid_composition_is_refused(modules, observers):
    with pytest.raises(ValueError):
        Brain.compose(2, 2, modules=modules, observers=observers)


@pytest.mark.parametrize(
    "controls",
    [{"budget": -1}, {"budget": True}, {"tolerance": float("nan")}, {"tolerance": "small"}],
)
def test_empty_imagined_branch_still_validates_solver_controls(controls):
    brain = Brain.compose(2, 2, modules=(4,))
    with pytest.raises(ValueError):
        brain.imagine([], **controls)


@pytest.mark.parametrize("options", [{"seed": True}, {"seed": -1}, {"episodic": "yes"}])
def test_invalid_composition_options_are_refused(options):
    with pytest.raises(ValueError):
        Brain.compose(2, 2, modules=(4,), **options)


def test_learning_surprise_gene_teaches_only_the_rows_the_brain_answers_differently() -> None:
    brain = Brain.compose(4, 2, modules=(8,), seed=0, learning_surprise=True)
    assert brain.learner.config.surprise is True
    x = np.random.default_rng(0).random((3, 4))
    labels = brain.predict(x)
    brain.step(x, teacher=labels)
    report = brain.last_learning
    assert report["demonstrations"] == 3.0
    assert report["demonstration_accepted_presentations"] == 0.0
    assert report["demonstration_agreed_presentations"] == 3.0
    brain.reset()
    brain.step(x, teacher=1 - labels)
    report = brain.last_learning
    assert report["demonstration_accepted_presentations"] == 3.0
    assert report["demonstration_agreed_presentations"] == 0.0
    assert Brain.compose(4, 2, modules=(8,), seed=0).learner.config.surprise is False


def test_learning_surprise_judges_from_the_brains_own_answer(monkeypatch) -> None:
    brain = Brain.compose(4, 2, modules=(8,), seed=0, learning_surprise=True)
    x = np.random.default_rng(1).random((3, 4))
    labels = np.array([0, 1, 0])
    read = []

    def choices(state):  # the answer the act would read, here fixed by the test
        read.append(len(state.activation))
        return np.array([0, 1, 1])

    monkeypatch.setattr(brain, "_choices", choices)
    brain.step(x, teacher=labels)
    assert read == [3]  # the judgment settle; the act's choice is the basal ganglia's
    assert brain.last_learning["demonstration_accepted_presentations"] == 1.0
    assert brain.last_learning["demonstration_agreed_presentations"] == 2.0
    assert brain.last_learning["demonstration_free_row_sweeps"] == (
        brain.last_learning["demonstration_free_steps"]
    )
