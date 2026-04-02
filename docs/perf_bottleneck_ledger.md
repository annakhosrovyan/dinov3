# Performance Bottleneck Ledger

This file tracks the current causal model of step time.

Use it to answer:

1. What is currently exposed on the critical path?
2. What type of bottleneck is it?
3. Which knobs can plausibly move it?
4. Which tempting ideas should be deferred?

## Current Summary

Current dominant bottleneck: `unknown`

Last updated from:

- run:
- date:
- artifacts:

Current decision:

- complete Step 2
- then collect the first baseline profiling trace

## Ledger Template

For each meaningful exposed cost, record a block like this:

```text
Name:
Evidence:
Approx Exposed Share:
Bound Type:
Candidate Knobs:
Prerequisites:
Why It Might Matter:
Why It Might Not Matter:
Decision:
```

## Active Ledger

