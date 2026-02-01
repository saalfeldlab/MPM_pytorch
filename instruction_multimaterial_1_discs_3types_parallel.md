# Parallel Mode Addendum

This file supplements the main instruction file. All rules from the main instructions apply.

## Parallel Execution

- **4 configs are trained simultaneously** per batch (4 slots: slot 00-03).
- Each slot has its own config file and analysis.log.
- All slots share the same analysis.md, memory.md, ucb_scores.txt, and reasoning.log.

## Config Editing Rules

1. **NEVER change the `dataset` field** in any slot config. Each slot has a unique dataset name that must be preserved.
2. **Edit all 4 config files** in a single analysis turn. Each slot should test a DIFFERENT mutation.
3. **Diversity requirement**: Do not duplicate the same mutation across slots. Use the 4 slots to explore different parameter dimensions or directions simultaneously.

## Analysis Format

- Write one `## Iter N:` entry per successful slot to the full log and memory file.
- Each entry follows the same format as the main instructions (Node, Mode/Strategy, Config, Metrics, etc.).
- Use iteration numbers from UCB scores as parent references.

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **principle-test** | Randomly pick one Established Principle from `memory.md` and design an experiment that tests or challenges it (see below) |

You may deviate from this split based on context (e.g., all exploit if early in block, all boundary-probe if everything converges).

### Slot 3: Principle Testing

At each batch, slot 3 should be used to **validate or challenge** one of the Established Principles listed in the working memory:

1. Read the "Established Principles" section in memory.md
2. **Randomly select one principle** (rotate through them across batches — do not repeat the same one consecutively)
3. Design a config that specifically tests this principle:
   - If the principle says "X works when Y", test it under a different condition
   - If the principle says "Z always fails", try to make Z succeed
   - If the principle gives a range, test at the boundary
4. In the log entry, write: `Mode/Strategy: principle-test`
5. In the Mutation line, include: `Testing principle: "[quoted principle text]"`
6. After results, update the principle's evidence level in memory.md:
   - Confirmed → keep in Established Principles
   - Contradicted → move to Open Questions with note

If there are no Established Principles yet (early in the experiment), use slot 3 as a **boundary-probe** instead.

## Parent Selection Rules

- UCB scores cover all iterations (sequential + parallel) in the current block.
- When selecting parents for next batch, choose from the highest UCB nodes — spread across different branches for diversity.
- At block boundaries, UCB resets as usual.

**CRITICAL — Preventing circular parent references**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. Within a parallel batch, `P` must ONLY reference:

1. **Your own id or an earlier slot** in the current batch (OK)
2. **Any iteration from a previous batch** (OK)

**NEVER** set `Next: parent=P` where P > your own id. This creates circular references because the tree builder applies `Next:` from iter N to override iter N+1's parent. If iter 2 writes `Next: parent=3`, iter 3 becomes its own parent.

**Example — batch iterations 1-4:**
- Iter 1 (id=1): `Next: parent=1` ← OK (self-reference, recommends itself)
- Iter 2 (id=2): `Next: parent=3` ← **BAD!** Creates circular: iter 3's parent = 3 (itself)
- Iter 3 (id=3): `Next: parent=3` ← OK (self-reference)
- Iter 4 (id=4): `Next: parent=3` ← OK (earlier slot)

**Simple rule: `Next: parent=P` must satisfy P ≤ your own id** (never a forward reference within the batch).

**Example — batch iterations 131-134:**

Valid parent references:
- Any iteration ≤ 130 (from previous batches)
- Iterations 131-134 where P ≤ current id
- NEVER reference iteration 135+ (future batch)

## Block Boundaries

- A block = `n_iter_block` iterations (default 8).
- Block boundaries may fall in the middle of a batch. When `>>> BLOCK END <<<` appears, perform block-end duties (update instructions, choose next block config, update memory) as specified in the main instructions.
