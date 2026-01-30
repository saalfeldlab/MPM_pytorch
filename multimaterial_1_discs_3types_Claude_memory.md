# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1     | siren_txy | Jp   | 100      | 0.996   | 0.976      | 4E-5             | 512 (or 384 speed) | 3                  | 5-10            | 200000              | 15.4 (or 12.2)      | omega_f=[5-10] optimal, lower than prior (15). 384×3 is speed Pareto. |
| 2     | siren_txy | F    | 100      | 0.998   | 0.998      | 4E-5 to 6E-5     | 256                | 4                  | 12              | 150000              | 8.1                 | omega_f=12 sharp optimum. Wide lr tolerance. Capacity ceiling at 256. |
| 3     | siren_txy | C    | 100      | 0.994   | 0.989      | 2E-5             | 640                | 3                  | 25              | 150000              | 15.7                | omega_f=25 unchanged from prior. Lower lr=2E-5 optimal. Capacity ceiling at 640. |
| 4     | siren_txy | S    | 100      | 0.729   | 0.824      | 2E-5             | 1280               | 3                  | 48              | 300000              | 166.2               | S depth=3 (NOT 4). omega_f=48 (no lower shift). R²=0.729 ceiling. |
| 5     | siren_txy | F    | 200      | 0.9997  | 0.9995     | 5E-5             | 256                | 4                  | 9-10            | 300000 (200k speed)  | 32.4 (27.6 speed)   | omega_f shifts 12→9 at 200f. lr narrows to 5E-5. Period params must stay 1.0. |

### Established Principles
1. **omega_f for Jp@9000particles**: Optimal at [5-10], LOWER than prior (15). More particles → lower omega_f.
2. **omega_f for F@9000particles**: Optimal at 12 (100f) or 9-10 (200f). SHARP optimum. More frames → lower omega_f.
3. **omega_f for C@9000particles**: Optimal at 25 - SAME as prior. C does NOT follow lower omega_f trend.
4. **omega_f for S@9000particles**: Optimal at 48 - similar to prior (50). S does NOT follow lower omega_f trend. Pattern: high-complexity fields (C, S) maintain omega_f, low-complexity (Jp, F) shift lower on more particles.
5. **n_layers**: Jp, C, S require EXACTLY 3 layers. F requires EXACTLY 4 layers.
6. **Speed Pareto for Jp**: 384×3@omega=10 achieves R²=0.995 in 12.2min.
7. **lr for Jp**: 4E-5 optimal, 5E-5 regresses.
8. **lr for F@100f**: WIDE tolerance [4E-5, 6E-5]. **lr for F@200f**: 5E-5 optimal (narrower tolerance).
9. **lr for C**: 2E-5 optimal. LOWER than Jp/F.
10. **lr for S**: 2E-5 hard-locked. lr=3E-5 catastrophic.
11. **F capacity ceiling**: 256×4 saturates F at both 100f and 200f. 384×4 severely overparameterized (0.970 vs 0.9997).
12. **C capacity ceiling**: 640×3 saturates C.
13. **S capacity**: 1280×3 required. No speed Pareto.
14. **Field-specific architectures**: Each field requires DIFFERENT optimal config on same dataset.
15. **Field difficulty ranking (this dataset)**: F(0.998+) > Jp(0.996) > C(0.994) >> S(0.729). Same ordering as prior.
16. **F data scaling**: No diminishing returns from 100→200 frames when omega_f re-tuned (12→9). Consistent with prior.
17. **Period parameters for F**: nnr_f_T_period=1.0 and nnr_f_xy_period=1.0 MANDATORY. T_period=2.0 causes catastrophic degradation (R²=0.790). xy_period=2.0 causes significant degradation (R²=0.987). Temporal smoothing 6× more damaging than spatial.
18. **F omega_f-frames scaling**: omega_f decreases ~2-3 per 100-frame increase on this dataset.
19. **lr-frames scaling for F**: More frames → narrower lr tolerance. 100f: [4-6]E-5 viable. 200f: only 5E-5 optimal.

### Open Questions
1. Does Jp maintain R²≈0.996 at 200 frames? Does omega_f=[5-10] hold or shift lower?
2. Would siren_id outperform siren_txy for any field on this dataset?
3. Can code modifications (loss function, scheduler) break S ceiling above 0.73?
4. Does C hurt from more data on this dataset (as it did on prior)?
5. Does Jp lr=4E-5 hold at 200 frames or shift (like F shifted to 5E-5)?

---

## Previous Block Summary (Block 5)

Block 5 (F@200frames@9000p): FULLY CHARACTERIZED. Best: R²=0.9997 with 256×4@omega=9@lr=5E-5@300k steps, 32.4min. Speed Pareto: 200k steps (R²=0.9988, 27.6min). omega_f shifted from 12→[9-10] confirming "more frames → lower omega_f". lr narrowed to 5E-5 only. Capacity ceiling at 256 re-confirmed. Period params must stay at 1.0 — T_period=2.0 catastrophic (0.790), xy_period=2.0 significant degradation (0.987). Branching rate: 25%.

---

## Current Block (Block 6)

### Block Info
Field: Jp, INR: siren_txy, n_frames: 200, n_particles: 9000
Iterations: 61-72

### Hypothesis
Jp@100frames achieved R²=0.996 with 512×3@omega=7@lr=4E-5@200k steps. Test data scaling to 200 frames:
1. Does Jp@200 maintain R²≈0.996? Prior dataset says beneficial but diminishing returns.
2. Does omega_f=[5-10] hold at 200f? F shifted from 12→9 at 200f, so Jp may shift from 7→5.
3. Does 512×3 capacity suffice or need increase?
4. Does lr=4E-5 hold or need adjustment? F shifted to 5E-5 at 200f.
5. Prior: Jp needs 2000 steps/frame → 400k steps at 200f. But overfitting risk above 2500 steps/frame.

Start: 512×3, omega_f=5, lr=4E-5, total_steps=400000 (2000 steps/frame), batch_size=1.

### Iterations This Block

**Iter 61: good** (R²=0.995, slope=0.978, 67.8min)
Node: id=61, parent=root
Config: 512×3, omega_f=5, lr=4E-5, steps=400k, batch=1, 200 frames
Mutation: Block baseline — Jp@200f with omega_f=5
Observation: Jp@200f maintains R²=0.995 (vs 0.996 at 100f). omega_f=5 viable. Training time 67.8min — long.
Next: parent=61

**Iter 62: good** (R²=0.994, slope=0.963, 67.7min)
Node: id=62, parent=61
Config: 512×3, omega_f=7, lr=4E-5, steps=400k, batch=1, 200 frames
Mutation: [omega_f]: 5.0 → 7.0
Observation: omega_f=7 mild regression from omega_f=5 (R²=0.994 vs 0.995). At 200f, lower omega_f slightly better.
Next: parent=62

**Iter 63: good** (R²=0.988, slope=0.934, 46.1min)
Node: id=63, parent=62
Config: 512×3, omega_f=7, lr=4E-5, steps=300k, batch=1, 200 frames
Mutation: [total_steps]: 400000 → 300000
Observation: Steps 400k→300k drops R² from 0.994→0.988. 1500 steps/frame insufficient for Jp@200f. Saves 21.6min but costs quality.
Next: parent=63

**Iter 64: good** (R²=0.992, slope=0.946, 54.3min)
Node: id=64, parent=63
Config: 512×3, omega_f=7, lr=5E-5, steps=300k, batch=1, 200 frames
Mutation: [learning_rate_NNR_f]: 4E-5 → 5E-5
Observation: lr 4E-5→5E-5 improves R² 0.988→0.992 at 300k steps. Higher lr partially compensates for fewer steps.
Next: parent=64

**Iter 65: good** (R²=0.992, slope=0.960, 46.1min)
Node: id=65, parent=64
Config: 512×3, omega_f=5, lr=5E-5, steps=300k, batch=1, 200 frames
Mutation: [omega_f]: 7.0 → 5.0
Observation: omega_f 7→5 at 300k/lr=5E-5: R² ties (0.992) but slope improves (0.946→0.960). omega_f=5 wins on magnitude fidelity.
Next: parent=65

**Iter 66: good** (R²=0.992, slope=0.950, 49.5min)
Node: id=66, parent=65
Config: 512×3, omega_f=3.0, lr=5E-5, steps=300k, batch=1, 200 frames
Mutation: [omega_f]: 5.0 → 3.0 (failure-probe)
Observation: omega_f=3 failure-probe — R² unchanged (0.992), slope slightly worse (0.960→0.950). Lower boundary NOT found. omega_f extremely flat [3-7] at 300k steps.
Visual: GT/Pred spatial match good. Scatter spread at GT>1.5. Loss still trending down at 300k.
Next: parent=66

**Iter 67: good** (R²=0.995, slope=0.965, 55.4min)
Node: id=67, parent=66
Config: 512×3, omega_f=3.0, lr=5E-5, steps=400k, batch=1, 200 frames
Mutation: [total_steps]: 300000 → 400000
Observation: omega_f=3+400k yields R²=0.995 — MATCHES omega_f=5+400k (iter 61). Confirms omega_f insensitivity at 400k too. Slope=0.965 (vs 0.978 for omega_f=5). Step count is dominant factor.
Visual: Loss still trending down at 400k. GT/Pred spatial match good. Spread at GT>1.5 in scatter. Early frames have higher per-frame MSE.
Next: parent=67

### Emerging Observations
- Jp@200f baseline: R²=0.995 with omega_f=5 at 400k steps. Quality maintained from 100f.
- omega_f map@400k: 3(0.995/slope=0.965) ≈ 5(0.995/slope=0.978) > 7(0.994/slope=0.963). Flat omega_f response [3-7]. omega_f=5 wins on slope.
- omega_f map@300k: 3(0.992/slope=0.950) ≈ 5(0.992/slope=0.960) ≈ 7(0.992/slope=0.946). Also flat. omega_f=5 gives best slope.
- Steps/frame map: 2000(0.994-0.995) > 1500(0.988-0.992). Jp@200f needs 2000 steps/frame for R²>0.99.
- Training time: 400k=55-68min, 300k=46-54min. Speed Pareto: 300k achieves R²=0.992 in ~46min.
- Slope=0.934-0.978 — persistent mild underprediction across all configs. Best slope: omega_f=5+lr=4E-5+400k (0.978).
- lr at 300k steps: 5E-5(0.992) > 4E-5(0.988). Higher lr helps at reduced steps.
- lr at 400k steps: 5E-5(0.995/slope=0.965) vs 4E-5(0.995/slope=0.978). lr=4E-5 has BETTER slope at 400k.
- 7 consecutive R²≥0.95. omega_f [3-7] all viable. omega_f lower boundary NOT found.
- Key insight: omega_f is NOT the differentiator. Step count (400k vs 300k) is dominant factor for R²>0.995. Loss still trending down at 400k — model not fully converged.
- Next: exploit Node 67 — lr 5E-5→6E-5 at 400k steps. Test if higher lr speeds convergence to push R²>0.995. Block 1 found Jp lr boundary at 6E-5 optimal (this dataset at 100f).
