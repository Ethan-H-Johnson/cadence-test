"""The episodic hippocampus of the composed brain (0.83.0): records vote in the choice."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def episodic(*args, **options) -> cd.Brain:
    """``Brain.compose`` with the episodic memory, unless ``memory_choice`` is named."""
    return cd.Brain.compose(*args, **{"memory_choice": True, **options})


def one_hot(index: int, width: int, rows: int = 1) -> np.ndarray:
    x = np.zeros((rows, width))
    x[:, index] = 1.0
    return x


def place(brain: cd.Brain, x: np.ndarray, action: int) -> float:
    """The place record of an action: the hippocampus keyed by the observation."""
    return float(brain.hippocampus.recall(x)[0, action])


def vote(brain: cd.Brain, x: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
    """The records' vote on the observations, for the stream's present context."""
    if key is None and brain.context_memory is not None:
        key = brain._episode_key(x)
    return brain._vote(x, key)


def consolidated(brain: cd.Brain, x: np.ndarray, action: int) -> float:
    """The shared, consolidated place record: a read in another batch size than the stream's."""
    return float(brain.hippocampus.recall(np.repeat(x, 3, axis=0))[0, action])


def test_the_episodic_memory_is_an_opt_in_preset_of_the_composed_brain():
    assert cd.Brain.compose(3, 2, seed=1).describe()["episodic"] is None
    brain = episodic(3, 2, seed=1)
    assert brain.hippocampus.rate == 0.5
    described = brain.describe()["episodic"]
    assert {key: described[key] for key in ("choice", "context", "replay", "demonstrations")} == {
        "choice": True, "context": 0.5, "replay": 8, "demonstrations": 1.0
    }
    np.testing.assert_array_equal(brain.hippocampus.pre, brain.sensory_index)
    np.testing.assert_array_equal(
        brain.context_memory.pre, np.concatenate([brain.sensory_index, brain.motor_index])
    )
    for memory in (brain.hippocampus, brain.context_memory):
        assert memory.separator.expansion == 1024 and memory.separator.winners == 24
    assert episodic(3, 2, seed=1, memory_context=0.0).context_memory is None
    released = cd.Brain.compose(3, 2, seed=1)
    assert released.describe()["episodic"] is None and released.context_memory is None
    np.testing.assert_array_equal(released.hippocampus.pre, released.sensory_index)
    assert released.hippocampus.separator is None and released.hippocampus.rate == 1.0
    for field in ("pre", "post", "sign"):
        np.testing.assert_array_equal(
            getattr(brain.connectome, field), getattr(released.connectome, field)
        )
    assert cd.Brain.compose(3, 2, seed=1, episodic=False).hippocampus is None


def test_the_episodic_settings_are_validated_construction_settings():
    with pytest.raises(ValueError, match="memory_context"):
        episodic(3, 2, memory_context=1.0)
    with pytest.raises(ValueError, match="memory_replay"):
        episodic(3, 2, memory_replay=-1)
    with pytest.raises(ValueError, match="memory_replay"):
        episodic(3, 2, memory_replay=True)
    with pytest.raises(ValueError, match="memory_choice must be boolean"):
        episodic(3, 2, memory_choice=1)
    with pytest.raises(ValueError, match="need memory_choice"):
        cd.Brain.compose(3, 2, memory_context=0.5)
    connectome = episodic(3, 2, seed=2).connectome
    with pytest.raises(ValueError, match="needs the episodic memory"):
        cd.Brain(connectome, episodic=False, memory_choice=True)
    with pytest.raises(ValueError, match="construction setting"):
        episodic(3, 2).retune(memory_context=0.2)
    brain = episodic(3, 2, seed=2)
    brain.retune(memory_rate=0.5)
    assert brain.hippocampus.rate == brain.context_memory.rate == 0.5


def test_a_record_votes_in_greedy_and_sampled_choices():
    brain = episodic(3, 2, seed=3)
    x = one_hot(0, 3)
    for _ in range(3):
        brain.reset()
        brain.step(x, teacher=[1])
    brain.reset()
    votes = vote(brain, x)
    assert votes[0, 1] > votes[0, 0]
    assert int(brain.act(x, greedy=True)[0]) == 1
    state = brain.basal_ganglia.state
    plain = brain.basal_ganglia.probabilities(state)
    voted = brain.basal_ganglia.probabilities(state, bias=votes)
    assert voted[0, 1] > plain[0, 1]


def test_the_vote_is_the_mean_of_the_place_and_the_context_records():
    brain = episodic(3, 2, seed=3)
    x = one_hot(0, 3)
    brain.step(x, teacher=[1])
    key = brain._episode_key(x)
    expected = 0.5 * (brain.hippocampus.recall(x) + brain.context_memory.recall(key))
    np.testing.assert_allclose(brain._vote(x, key), expected)


def test_a_demonstration_outlives_a_zero_outcome_in_the_middle_of_an_attempt():
    brain = episodic(3, 3, seed=4, memory_context=0.0)
    start, middle = one_hot(0, 3), one_hot(1, 3)
    brain.step(start, teacher=[2])
    shown = place(brain, start, 2)
    assert shown > 0
    brain._moment = (start.copy(), np.array([2]))  # the shown action was the one executed
    brain.learn(np.zeros(1), np.zeros(1, bool), middle)
    assert place(brain, start, 2) == pytest.approx(shown)
    brain.step(start)  # decide again in the same situation
    brain._moment = (start.copy(), np.array([2]))
    brain.learn(np.zeros(1), np.ones(1, bool), middle)  # the attempt ended with nothing
    assert place(brain, start, 2) < shown


def test_only_the_streams_whose_outcome_is_known_write_their_records():
    brain = episodic(2, 2, seed=5)
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    brain.act(x)
    writes = brain.hippocampus.writes, brain.context_memory.writes
    brain.learn(np.array([1.0, 0.0]), np.zeros(2, bool), x)  # one paid, one not yet known
    assert (brain.hippocampus.writes, brain.context_memory.writes) == (
        writes[0] + 1, writes[1] + 1
    )
    brain.act(x)
    brain.learn(np.array([0.0, 0.0]), np.array([False, True]), x)  # one attempt ended
    assert brain.hippocampus.writes == writes[0] + 2


def test_replay_credits_the_path_of_a_rewarded_attempt_and_not_its_stalls():
    brain = episodic(4, 3, seed=6, memory_context=0.0, actor_gamma=0.9)
    plan = [(0, 1), (0, 0), (1, 2), (2, 2)]  # a stall at situation 0, then onwards
    for situation, action in plan:
        x = one_hot(situation, 4)
        brain.act(x)
        brain._paths[0][-1] = (None, np.array(action), x[0])
        brain._moment = (x, np.array([action]))
    assert [int(np.argmax(entry[2])) for entry in brain._paths[0]] == [0, 1, 2]
    assert [int(entry[1]) for entry in brain._paths[0]] == [0, 2, 2]
    stalled = place(brain, one_hot(0, 4), 1)
    writes = brain.hippocampus.writes
    brain.learn(np.ones(1), np.ones(1, bool), one_hot(3, 4))
    assert brain.hippocampus.writes == writes + 3  # the paid decision and two on its path
    assert place(brain, one_hot(0, 4), 0) > 0 and place(brain, one_hot(1, 4), 2) > 0
    assert place(brain, one_hot(0, 4), 1) == pytest.approx(stalled)  # nothing wrote action 1
    # discounted per step of the path: one step back holds more than two
    assert consolidated(brain, one_hot(1, 4), 2) > consolidated(brain, one_hot(0, 4), 0) > 0
    assert brain._paths[0] == []


def test_a_continuing_life_is_not_replayed():
    brain = episodic(4, 2, seed=7, memory_context=0.0)
    brain.act(one_hot(0, 4))
    brain.learn(np.zeros(1), np.zeros(1, bool), one_hot(1, 4))
    brain.act(one_hot(1, 4))
    writes = brain.hippocampus.writes
    brain.learn(np.ones(1), np.zeros(1, bool), one_hot(2, 4))  # paid, but no attempt ended
    assert brain.hippocampus.writes == writes + 1  # the paid decision alone
    assert len(brain._paths[0]) == 2


def test_the_temporal_context_recalls_a_cue_that_is_gone():
    brain = episodic(3, 2, seed=8)
    cues = {0: one_hot(0, 3), 1: one_hot(1, 3)}
    go = one_hot(2, 3)
    for cue, shown in cues.items():  # one demonstration of each cue's answer
        brain.reset()
        brain.step(shown, teacher=[cue])
        brain.step(go, reward=[0.0], teacher=[cue])
    for cue, shown in cues.items():
        brain.reset()
        brain.act(shown, greedy=True)
        assert int(brain.act(go, greedy=True)[0]) == cue
    # the place records alone cannot tell the two moments at go apart
    places = episodic(3, 2, seed=8, memory_context=0.0)
    assert places.context_memory is None and places._context is None


def test_what_the_stream_did_enters_its_context():
    brain = episodic(2, 3, seed=9)
    x = one_hot(0, 2)
    action = int(brain.act(x)[0])
    context = brain._context[0]
    assert context[2 + action] == pytest.approx(0.5)
    assert context[0] == pytest.approx(0.5)
    brain.learn(np.zeros(1), np.ones(1, bool), x)
    assert not brain._context.any() and brain._paths == [[]]


def test_a_saved_episodic_brain_continues_exactly(tmp_path):
    brain = episodic(3, 2, seed=10, arousal=True)
    for t in range(4):
        brain.step(one_hot(t % 3, 3), reward=None if t == 0 else [0.0])
    path = brain.save(tmp_path / "brain.npz")
    copy = cd.Brain.load(path)
    assert copy.describe()["episodic"] == brain.describe()["episodic"]
    np.testing.assert_array_equal(copy._context, brain._context)
    np.testing.assert_array_equal(copy._decision, brain._decision)
    assert len(copy._paths[0]) == len(brain._paths[0])
    for t in range(6):
        reward, done = [float(t == 5)], [t == 5]
        np.testing.assert_array_equal(
            brain.step(one_hot(t % 2, 3), reward=reward, done=done),
            copy.step(one_hot(t % 2, 3), reward=reward, done=done),
        )
    for name in ("hippocampus", "context_memory"):
        np.testing.assert_allclose(
            getattr(copy, name).consolidated, getattr(brain, name).consolidated
        )
    meta = json.loads(str(np.load(path)["generic"]))
    assert meta["format"] == "cadence-generic/6"
    places = episodic(3, 2, seed=10, memory_context=0.0)
    places.step(one_hot(0, 3))
    resumed = cd.Brain.load(places.save(tmp_path / "places.npz"))
    assert resumed.context_memory is None and resumed.memory_choice
    released = cd.Brain.compose(3, 2, seed=10)
    released.act(one_hot(0, 3))
    meta = json.loads(str(np.load(released.save(tmp_path / "released.npz"))["generic"]))
    assert meta["format"] == "cadence-generic/2" and "episodic" not in meta


def test_a_calm_brain_records_the_outcome_that_woke_it():
    brain = episodic(2, 2, seed=11, arousal={"youth": 0})
    x = one_hot(0, 2)
    brain.live(x)
    for _ in range(5):  # outcomes as forecast: routine
        brain.live(x, reward=[0.0])
    assert brain.last_arousal["mode"] == "routine"
    action, key = int(brain._lived[1][0]), brain._decision.copy()
    before = vote(brain, x, key)[0, action]
    brain.live(x, reward=[5.0])  # far beyond the forecast: it wakes the calm brain
    assert brain.last_arousal["recorded"]
    assert vote(brain, x, key)[0, action] > before


def test_waiting_drifts_the_context():
    brain = episodic(3, 2, seed=12, arousal=True)
    brain.live(one_hot(0, 3))
    before = brain._context.copy()
    brain.wait(one_hot(1, 3))
    np.testing.assert_allclose(brain._context, 0.5 * before + 0.5 * np.r_[one_hot(1, 3)[0], 0, 0])


def test_slots_take_the_records_vote_per_slot():
    brain = episodic(2, 4, slots=2, seed=13)
    x = one_hot(0, 2)
    for _ in range(3):
        brain.reset()
        brain.step(x, teacher=[[1, 0]])
    brain.reset()
    np.testing.assert_array_equal(brain.act(x, greedy=True)[0], [1, 0])


def test_a_choosing_memory_is_not_read_into_the_drive(monkeypatch):
    brain = episodic(3, 2, seed=14)
    brain.step(one_hot(0, 3), teacher=[1])

    def refused(*args, **kwargs):
        raise AssertionError("the records were read into the drive")

    for memory in (brain.hippocampus, brain.context_memory):
        monkeypatch.setattr(memory, "stimulate", refused)
    brain.stimulus(one_hot(0, 3))
    brain.imagine([one_hot(0, 3), one_hot(1, 3)])
    brain.act(one_hot(0, 3))
