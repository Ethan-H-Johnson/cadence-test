# Episodic memory: what happened in this situation before

A brain's synapses learn slowly. Each update moves the whole policy a little, and neighbouring
situations share most of their representation, so one lesson barely moves one answer. Animals
also learn in one go: a place where food was found once, a route a parent showed once, the
step of a song that comes next. Since 0.83.0 the composed brain can keep such episodes in its
hippocampus, an episodic memory beside the slow cortex whose records vote in every choice:
`Brain.compose(..., memory_choice=True)`. It is not yet the default; [when to turn it
on](#when-to-turn-it-on) says why.

## How it works

```text
observation now ─────────────────────► place records ───┐
seen and done before ─┐                                  ├─► mean ─► vote ─┐
observation now ──────┴─► temporal context ─► context records ─┘           ├─► softmax ─► action
observation now ─► settle ─► motor readout ─────────────────────────────────┘

outcome, when known ──────────► record of the chosen action under its key
attempt ended in reward ───────► replay: the decisions on its path, recorded again backwards
```

- **Two keys for every record.** Each record is kept twice. The place records are keyed by
  the present observation. The context records are keyed by the temporal context: the
  present observation together with a fading trace of what the stream saw and did before,
  past observations and the brain's own issued actions. `memory_context` is the share the
  past keeps from one moment to the next; the present observation enters with the rest.
- **Why two.** Two moments that look alike but follow different histories get different
  context keys, so a cue that is gone still shapes the vote, and so does the place in a
  sequence of the brain's own actions. Where history does not matter, the context splits one
  situation's records over every path that reaches it; the place records keep them together,
  so a known situation reached along a new path is still recognized. The trace is fixed,
  not learned: a record stays addressable while the cortex learns.
- **Records** hold one value per action for each key, in reward units. Keys are separated
  first (a fixed random expansion to 1,024 cells, of which the 24 most active stay), so
  different situations write different synapses.
- **What is written.** The outcome of the chosen action, when it is known: a reward or a
  punishment, the end of an attempt (`done`), or a surprise that woke a calm `live` brain. A
  zero outcome in the middle of an attempt is not yet an outcome and leaves the record as it
  was. A caregiver's demonstration (`teacher=`) is recorded for the situation it was shown in,
  at `memory_demonstrations`, as if the shown action had paid that much. A write moves the
  record halfway toward the outcome (`memory_rate` 0.5): one outcome is remembered at once,
  and one misleading outcome cannot erase what repeated outcomes established.
- **Replay.** When an attempt ends in reward, the decisions on its path are recorded again,
  backwards, at the reward discounted by the actor's `gamma` per step, up to `memory_replay`
  steps. The path keeps one decision per situation, the one that left it: an action that left
  the situation unchanged is not on the path and is not credited. A continuing life without
  `done` is not replayed; there, a reward is recorded for the decision that earned it and
  earlier decisions are credited by the actor's eligibility.
- **The vote.** At every choice, greedy or sampled, the mean of the place and the context
  records is added to the motor readout before the softmax. The slow cortex keeps learning from the actions the
  brain takes, so what the memory chose and what paid is consolidated into the synapses over
  time.
- **Two speeds.** The store has the per-stream fast residual and the shared consolidated
  matrix of [`SynapticMemory`](memory.md#memory-in-the-composed-brain): a record is at full
  strength at once in its stream and is consolidated with repetition and salience
  (`consolidation`).

## Turning it on

```python
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, arousal=True, memory_choice=True)
brain.step([[1.0, 0.0, 0.0]], teacher=[0])                  # shown once: remembered
action = brain.live([[1.0, 0.0, 0.0]])                      # the record votes for 0
```

`memory_choice=True` brings the settings below. Each can be passed beside it.

| Setting | With `memory_choice=True` | What it does |
| --- | --- | --- |
| `memory_choice` | `True` | The records vote in the choice; the founder `False` keeps the hippocampus of earlier releases, keyed by the senses and read into the motor drive |
| `memory_context` | `0.5` | The share of the past in the context key; 0 keeps the place records alone |
| `memory_replay` | `8` | How many decisions back an attempt that ended in reward is replayed |
| `memory_demonstrations` | `1.0` | The value at which a shown action is recorded; 0 records none |
| `memory_rate` | `0.5` | How far one write moves a record toward the outcome; 1.0 lets the last outcome replace it |
| `memory_expansion`, `memory_winners` | `1024`, `24` | Separation of the keys; 0 cells use the key as it is |

These are construction settings: pass them to `Brain.compose(...)`; `retune` refuses them,
except `memory_rate`, a store setting like `memory_decay`, `memory_amplitude` and
`consolidation`, which `retune` changes for both stores together. The same settings turn it
on in `Brain(...)` and `Brain.build(...)`, where every setting has to be named.

## When to turn it on

Turn it on when a skill is taught by showing or correcting, when a chain of actions is paid at
its end and the world marks the end of each attempt, when a cue that is gone must still decide
an answer, and for sequences of the brain's own actions. Measured on fresh founders against the
composition without it (development tasks, never used to choose the settings):

| Task (fresh founders) | Without | With `memory_choice=True` |
| --- | --- | --- |
| Four-step crafting chain, a caregiver corrects executed mistakes (5 founders): success after teaching | 41.6% | 94.4% |
| the same, then 80 episodes alone through `live` | 38.0% | 99.2% |
| the same chain from reward alone, through `step`, then alone | 15.8% | 79.0% |
| A second crafting chain (5 seeds): taught, after the caregiver leaves | 33.4% | 99.4% |
| the same from reward alone, after 80 episodes alone | 20.0% | 70.2% |
| Delayed response learned from reward (6 seeds) | | 6 of 6 at 1.0 |
| Continuing bandit whose history does not matter (6 seeds), hit rate | 0.73 | 1.00 |
| Two four-note songs cued once, 10 lessons (6 seeds): notes right, whole songs | 0.60, 1 of 12 | 0.79, 4 of 12 |
| Taught recall of a vanished cue (4 founders): clean-4, clean-8, distractor-2, capacity-2 | 0.55 | 0.74 |

The recall gain is uneven: two founders on which recall was weak gained 0.37 and 0.49; on one
whose recall was already strong (0.875) it fell to 0.68, where order (which of two cues came
last) rose from 0.00 to 1.00.

Not yet in a continuing life whose rules reverse: in the odour nursery's confirmation lives,
three of twenty reversed or returned incompletely (their last hundred actions 76 to 82%
optimal), and one of them also lost its stable pair in a third of its probes during one rule,
where the composition without it passed every gate. A record of a punished approach keeps voting against the odour that now
pays. In a continuing life without `done`, no attempt is replayed: on the prerequisite-chain
nursery's observed chain, paid only at the completion, the completing action's record spread to
neighbouring stages and lowered the stage accuracy from 0.67 to 0.33.

## Settings

| Setting | Raise it when | Lower it when |
| --- | --- | --- |
| `memory_context` | the answer depends on what came long before: a cue that is gone, the next step of a long sequence of the brain's own actions | history does not matter, and the context records only dilute the place records (0 keeps the place records alone) |
| `memory_replay` | rewards come at the end of long chains of decisions | (0 replays nothing) |
| `memory_demonstrations` | shown actions should outweigh the rewards of the brain's own trials | the caregiver is unreliable (0 records no demonstration) |
| `memory_expansion` | many situations must be told apart | (0 uses the key unseparated) |
| `memory_rate` | the world changes quickly and the latest outcome is the best guide | outcomes are noisy, or one failure should not undo a learned skill |

Mark the end of an attempt with `done=True` where the world has one: a chain completed, a
trial over, a game lost. The end makes the outcome known, starts the next attempt with a fresh
context and lets a rewarded attempt be replayed.

## Check that the memory does the work

Compare the same acquired brain with and without its records, on copies:

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "brain.npz")
brain.save(path)
remembering, forgetting = Brain.load(path), Brain.load(path)
forgetting.hippocampus.clear()  # erase the place records
forgetting.context_memory.clear()  # and the context records
x = [[1.0, 0.0, 0.0]]
print(remembering.act(x, greedy=True), forgetting.act(x, greedy=True))
```

If the answers differ, the records decide that choice. If they never differ, the slow
cortex has consolidated what the memory held, or the memory holds nothing for that key.

## What it is not

- It is not a list of episodes. Its storage is fixed by the key width and the separation; it
  does not grow with experience. Correlated keys interfere, as every delta-rule store does.
- The keys are not learned. The memory recalls what happened in the same place and in the
  same context; it does not discover which part of the history matters. Where it matters,
  the place records of the aliased moments average over histories and halve the context's
  vote; where it does not, the context records add little.
- Records are in reward units. With rewards of a very different size, the vote weighs more or
  less against the readout; `memory_amplitude` scales the read.
- `predict` and `fit` answer independent samples and read no memory.

See [context](context.md) for the carried interpretation of the working trace,
[memory](memory.md) for the stores and their rules, and [sequential tasks](sequential-tasks.md)
for chains of actions.
