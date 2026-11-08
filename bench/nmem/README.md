# Pruning on a real corpus

The first experiment in this project with **real usage data** rather than a
synthetic benchmark. It answers a question the other benchmarks could not: when
you have to forget things, which policy keeps what mattered?

## The data

`~/.neuralmemory/brains/default.db`, copied and never opened for writing. 2925
fibers across months of use, of which:

```
2891  have usable text after dropping mojibake rows
 660  were ever recalled          (22.8%)
2231  never recalled
```

**That split is the thing no benchmark provides.** `frequency` is a real record
of which memories the user actually asked for, accumulated by a system that was
in daily use. It is not a synthetic label; it is what happened.

A share of the corpus is UTF-8 decoded as GBK and reads as `涓�` noise. Those rows
are excluded — they are unsearchable, so counting them would test nothing.

## Result

Prune to 660 memories — the size of the genuinely-used set.

```
strategy                 kept   lived kept   self-recall
---------------------- ------ ------------ -------------
no pruning               2891       100.0%         64.6%
by weight (usage)         660       100.0%         31.5%
by age (newest first)     660         0.0%         90.2%
random                    660        22.1%         69.8%
```

`lived kept` is the share of the 660 ever-recalled memories that survived.

**By weight keeps every single one. By age keeps none of them.**

The reason is not subtle once you see it: a memory that has been useful for
months is old. Pruning the oldest first deletes exactly the memories that earned
their keep, and it does so *worse than random* — 0.0% against 22.1% — because the
age ordering is anti-correlated with usefulness here, not merely uncorrelated.

## The number that argues the other way

`self-recall` — whether a memory is still findable by its own text — goes *down*
after weight-based pruning, from 64.6% to 31.5%. That looks like a contradiction
and it is worth stating plainly rather than burying.

The explanation is that the surviving set is the memories that were used, and
memories that were used repeatedly tend to be about the same few subjects. Keep
only those and they compete with each other: the one you want is still there, but
three near-identical ones are too. Age-based pruning keeps newer, more varied
memories, which are individually easy to locate and mostly useless.

So the two metrics measure different things:

- `lived kept` asks **did you keep what mattered** — weight wins, completely.
- `self-recall` asks **is each survivor individually findable** — age wins, and
  the memories it wins on are the ones nobody asked for.

Which one matters depends on what a memory store is for. If it exists to answer
questions the user actually asks, `lived kept` is the one. If it exists to be a
clean browsable archive, `self-recall` is. This project is the former.

## What this does not show

- **One user, one corpus, one machine.** 2925 fibers from a single person's
  workflow. The 0.0% is a property of *this* usage pattern, where useful memories
  skew old. Someone whose work is uniformly recent would see a different number.
- **`frequency` is nmem's counter, not ours.** It counts recalls under a
  different retrieval system with different failure modes. It is a proxy for
  usefulness, and a proxy is not a measurement.
- **Weight-based pruning was not tested against a human judgement of value.** The
  claim is only that it agrees with observed usage. Whether observed usage is the
  right definition of value is a separate question, and unanswered.

## Reproducing

```bash
python -m bench.nmem.prune_probe            # full corpus, ~2 min
python -m bench.nmem.prune_probe --limit 600
```

The database path can be overridden with `--db`. Nothing is written to the source
copy and no memory text is committed — only the numbers above.
