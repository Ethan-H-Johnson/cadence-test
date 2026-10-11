# Sequential tasks and prerequisite chains

A continuing brain that learns a single step well, food in one move, often fails a chain of
steps: gather wood, gather stone, make fire, warm up. The usual sign is a policy that
collapses onto the one action that pays most often. This page says what the repository's
measurements establish about that failure and what to do about it today. Every point names
the receipt behind it; none of it is a default change.

## What breaks a chain

**The chain's state is not in the observation.** If the brain cannot see what it carries,
the right action depends on history, and the task is a short-term memory task. The declared
recall chamber did not establish a reliable working-trace horizon on its fresh founders
([recall/5](../benchmarks/recall/README.md#recall5-on-fresh-founders-313-314-and-315-closure-failed-horizons-none-0-and-0),
[issue 84](https://github.com/muellerberndt/cadence/issues/84)). Those results do not establish
recall of a four-step inventory, or a universal upper bound on the trace under every task
and operating point. Without sufficient history, a policy cannot reliably distinguish
chain stages that have identical observations; it may settle on one action or keep sampling.
Expose inventory state when the creature can sense it: has wood, has stone, has fire.
Keep a hidden-inventory condition as a separate memory test.

**Pay arrives several choices later.** Credit for an earlier action across intervening
choices is the delayed key-door contract of
[issue 111](https://github.com/muellerberndt/cadence/issues/111). The
[key-door nursery](../benchmarks/keydoor/README.md) acquires at delays of two and five
choices and not at ten; at five the critic's rate was the bottleneck, and the chamber runs
its critic at 5.0 in place of the composed 0.3. Pay only at the end of a four-step chain sits
at the edge of what the eligibility (`actor_lam` 0.6, `actor_gamma` 0.95) reaches.

**The readout saturates.** Near saturation, a finite nudge can produce very little
activity contrast: the arena's nursery grids record motor probabilities at 0.98 that the aroused
sampling could not escape, and the key-door chamber records the same latch under punished
exploration. The readout's intrinsic plasticity (`learning_homeostasis_rate`, founder 0,
[learning](learning.md)) keeps a readout in range under teaching; under reward learning in
the [prerequisite-chain nursery](../benchmarks/chain/README.md) its development point (rate
0.1 toward 0.3) lowers every reading, because a six-action readout's mean activation sits
near 0.17 and a pull toward 0.3 flattens it. It is not a remedy for the collapse as declared.

**Learning continues at the acquisition step.** A policy acquired at the nursery's actor
step erodes under its own continued learning, in a melee and in the nursery itself
([issue 169](https://github.com/muellerberndt/cadence/issues/169), the
[competing-skill ring](../benchmarks/competing/README.md#the-arena-sixty-fights-2026-10-10)).
More drilling past the point of acquisition does not consolidate, it moves the policy. A
step ten to thirty times smaller after acquisition held the skill while learning went on.

## The pattern that the measurements support

1. **Put the chain's state in the observation.** Inventory flags are senses, not a
   workaround, until a measured memory horizon exists.
2. **Grade executed actions; do not label.** Let the brain act through `Brain.live` and pay
   the outcome. The world supplies problems, never answers. Label teaching of a sequence is a
   different contract ([issue 121](https://github.com/muellerberndt/cadence/issues/121)) and
   showed its own readout instability in the recall chamber.
3. **Compare reward schedules on the complete chain.** In the six-seed four-step
   development chamber, observed-state reward at each correct step reached 204
   greedy completions per thousand moments; end-only pay reached 42 and randomly
   interleaved starting stages reached 33. Granted prerequisites can let the brain
   meet its need before learning the complete chain. Backward lessons are therefore
   an experimental control, not an established remedy. Disclose supplied shaping
   rewards and verify the full chain from stage zero under end-only pay.
4. **Track acquisition and retention at a declared probe schedule.** Longer training
   can degrade an acquired skill. Select a training budget in development, then
   freeze it before testing fresh founders; disclose any checkpoint selection.
5. **Lower the actor step after acquisition.** The ring stage of the arena
   (`actor_eta` 0.001 to 0.003 against the nursery's 0.03) is the measured remedy for a
   skill that erodes under continued learning.
6. **Read the sampled policy beside the greedy one.** A frozen greedy copy can answer with
   one action in every situation while the sampled policy's probabilities move with the
   situation (the arena's fingerprint). Read the probability of the right action per
   situation, and the completion rate of the sampled policy, not only the argmax.
7. **Test the critic rate against its control.** The key-door nursery measured
   `actor_eta_critic=5.0` against the composed 0.3. This is a task-specific operating
   point, not evidence that raising the rate helps every delayed-credit task.

## What is not available yet

A memory horizon for a hidden chain state (the next lane of issue 84 is the write into the
working trace as an action of the basal ganglia under a reward at the query); a mechanism
that keeps an acquired skill at the nursery's own actor step (issue 169's next candidate is
a step that follows the brain's own uncertainty); and a passed gate for delayed credit
beyond five choices (issue 111). The
[prerequisite-chain nursery](../benchmarks/chain/README.md) is the instrument for this
page: the chain's state observed or hidden, pay at the end or in backward lessons, the
founder brain against uniform random, a frozen newborn, the stationary memoryless ceiling
and a tabular learner with the same information.
