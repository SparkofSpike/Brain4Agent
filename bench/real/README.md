# Real-store probe

The benchmarks in `bench/zh`, `bench/pmb` and `bench/locomo` use fixed fixtures.
A real store has thousands of short, same-topic memories and an agent that types
two or three keywords instead of a question. D-039 measured the difference: 0.95
hit@1 on the Chinese fixture, 0.27 for two-word queries on a 2898-memory store.

This probe keeps that second number visible. It derives queries from the store
itself and never writes back.

## Run

```bash
cp /path/to/brain.db /tmp/brain-copy.db
python -m bench.real.probe --db /tmp/brain-copy.db --limit 300
```

Run it on a copy: `search` increments access counters, so even a read writes.

## What the numbers mean

- `hit@1` — the memory the keywords came from ranked first.
- `hit@5` — the memory was in the first five.
- `by_len` — split by query length. Two-word queries are the hard case; if this
  number drops, the store got worse at the way an agent actually asks.

No memory text is printed and no fixture is committed. The probe is a ruler, not
a benchmark.
