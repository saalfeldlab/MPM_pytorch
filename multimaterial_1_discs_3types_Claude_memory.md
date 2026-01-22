# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1     | siren_txy | Jp   | 48       | 0.968   | 0.901      | 2.5E-5           | 512                | 3                  | 35.0            | 100000              | 8.8                 | omega_f=35 optimal, lr zone 2E-5→3E-5 |
| 2     | siren_txy | F    | 48       | 0.9995  | 0.9995     | 3E-5             | 256                | 4                  | 25.0            | 100000              | 6.3                 | omega_f=25, F easier than Jp, 256 sufficient |
| 3     | siren_txy | S    | 48       | 0.618   | 0.629      | 2E-5             | 512                | 4                  | 50.0            | 150000              | 16.7                | S HARD field, R²=0.618 ceiling, omega_f=50 sharp peak |
| 4     | siren_txy | C    | 48       | 0.993   | 0.981      | 3E-5             | 384                | 3                  | 30.0            | 100000              | 7.1                 | omega_f=30, C like F (easy), 384 optimal |
| 5     | siren_txy | F    | 100      | 0.9998  | 0.9999     | 3E-5             | 256                | 4                  | 15-25           | 100-150k            | 6.4                 | Data scaling SUCCESS, omega_f 15-25 plateau |
| 6     | siren_txy | Jp   | 100      | 0.982   | 0.938      | 3E-5             | 384                | 3                  | 30.0            | 200000              | 43.5                | Data scaling SUCCESS for Jp (+0.014 vs 48 frames), 384>512>256, 2000 steps/frame needed |
| 7     | siren_txy | S    | 48→100   | 0.708   | 0.735      | 2E-5             | 768                | 4                  | 50.0            | 250000              | 224.9               | Data scaling FAILS for S. Capacity scaling WORKS (768>512). NEW RECORD R²=0.708 at 48 frames |
| 8     | siren_txy | C    | 100      | 0.996   | 0.990      | 3-5E-5           | 384                | 3                  | 25-35           | 200000              | 43.4                | Data scaling WORKS for C (+0.003 vs 48 frames). lr tolerance [3E-5, 5E-5] widest. 256×3 Pareto: R²=0.993 in 10min |
| 9     | siren_txy | S    | 48       | 0.757*  | 0.817      | 2E-5             | 1280               | 4                  | 50.0            | 150000              | 57.9                | *UNRELIABLE: same config ranges R²=0.084-0.757. S field has EXTREME stochastic variance |
| 10    | siren_txy | Jp   | 200      | 0.989*  | 0.946      | 4E-5             | 384                | 3                  | 20.0            | 400000              | 25.0                | Data scaling SUCCESS (+0.007 vs 100 frames). *Variance ~0.012 (robustness: 0.977). Diminishing returns. |

### Established Principles
1. **omega_f sensitivity**: Field-specific optimal frequencies - Jp→20-25 (200 frames), F→15-25, C→25-35, S→50. More frames → lower optimal omega_f.
2. **lr optimal zone**: F→3E-5-4E-5, Jp→4E-5 (200 frames), C→3E-5-5E-5 (widest), S→2E-5 (strictest)
3. **hidden_dim field-dependent**: F needs 256, Jp/C need 384, S needs 768. Capacity ceiling varies by field.
4. **n_layers optimal**: F→4 layers, Jp/C/S→3-4 layers. Jp is depth-sensitive (4 layers degrades severely).
5. **Depth > width for F**: Adding depth more efficient than width for F field. Other fields prefer width.
6. **total_steps**: Field-dependent steps/frame ratio. F~1000, Jp~2000, S~3000+ steps/frame.
7. **batch_size**: Use batch_size=1 to avoid training time explosion
8. **Field difficulty ranking**: F (R²=0.9998) > C (R²=0.993) >> Jp (R²=0.982) >> S (R²=0.708). Stress tensor fundamentally harder.
9. **Local optimum detection**: When 5+ mutations from best node all regress, config-level is exhausted
10. **Data scaling benefit**: F and Jp improve with more frames. S does NOT (data scaling FAILS for S).
11. **LR-omega_f interaction**: LR tolerance widens at lower omega_f values
12. **Pareto-optimal configs**: F→256×4, Jp/C→384×3, S→768×4
13. **S field unique**: Higher capacity needed (768 vs 512), data scaling hurts, lr zone narrower (2E-5 only)
14. **STOCHASTIC VARIANCE (Block 9)**: S field at high capacity (1024+) has EXTREME variance - same config can produce R² from 0.08 to 0.76
15. **OVERFITTING LIMIT (Block 10)**: More training steps beyond optimal causes overfitting (Jp@200 frames: 400k OK, 500k regresses 0.989→0.939)
16. **DATA SCALING DIMINISHING RETURNS (Block 10)**: Jp data scaling gain: +0.014 (48→100 frames), +0.007 (100→200 frames). Halving each doubling.
17. **Jp MODERATE VARIANCE (Block 10)**: Jp has ~0.012 R² variance between identical runs - manageable unlike S field

### Open Questions
1. ~~Can C field improve beyond R²=0.993 with 100 frames?~~ **ANSWERED Block 8**: YES, R²=0.996 achieved with 2000 steps/frame
2. ~~Can S field improve beyond R²=0.708 with even more capacity (1024×4)?~~ **ANSWERED Block 9**: Yes R²=0.757, but UNRELIABLE (variance 0.08-0.76)
3. Can S field be stabilized with code modification? (Loss scaling, gradient clipping, normalization)
4. Can we exceed R²=0.9998 on F field with 200+ frames?
5. ~~Can Jp exceed R²=0.982 with 200 frames?~~ **ANSWERED Block 10**: YES, R²=0.989 achieved with omega_f=20, lr=4E-5
6. ~~Is there a data scaling ceiling for Jp/F/C?~~ **PARTIAL ANSWER Block 10**: Jp shows diminishing returns (halving gain per doubling frames)
7. **NEW: Can F field exceed R²=0.9998 with 200 frames?** (data scaling test)
8. **NEW: Can C field benefit from 200 frames?** (C untested beyond 100 frames)

---

## Previous Block Summary (Block 10)

**Field**: Jp (plastic deformation), **INR Type**: siren_txy, **n_frames**: 200
**Best achieved**: R²=0.989 (Node 115, lr=4E-5, omega_f=20, 384×3) - variance ~0.012 (robustness: 0.977)
**Key findings**: Data scaling SUCCESS (+0.007 vs 100 frames). Optimal omega_f shifted to 20-25 (from 30-35 at fewer frames). Overfitting detected at 500k steps. Depth (n_layers=4) and capacity (512) both HURT Jp.

---

## Current Block (Block 11)

### Block Info
Field: field_name=F, inr_type=siren_txy
Iterations: 121-132
n_training_frames: 200 (testing data scaling for F)

### Hypothesis
Testing data scaling for F field with 200 frames (2x increase from Block 5's 100 frames).
- Block 5 achieved R²=0.9998 with 100 frames at omega_f=15-25 plateau
- Prediction: 200 frames may push F toward R²=0.99999 or confirm diminishing returns like Jp
- Using Block 5 optimal config: lr=3E-5, hidden_dim=256, n_layers=4, omega_f=20, total_steps=200000 (1000 steps/frame)
- Goal: Test if F field shows same diminishing returns as Jp at higher frame counts

### Iterations This Block

## Iter 121: good (UNEXPECTED REGRESSION - need more steps)
Node: id=121, parent=root
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.935, final_mse=3.15E-2, slope=0.944, training_time=12.1min
Mutation: Block 5 optimal with 200 frames (1000 steps/frame)
Observation: R²=0.935 << 0.9998 (Block 5 @100 frames). 1000 steps/frame INSUFFICIENT. Need more steps.
Next: parent=121

## Iter 122: excellent (HYPOTHESIS CONFIRMED)
Node: id=122, parent=121
Mode/Strategy: success-exploit/exploit
Config: lr_NNR_f=3E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.9998, final_mse=7.75E-5, slope=0.9996, training_time=18.0min
Field: field_name=F, inr_type=siren_txy
Mutation: total_steps: 200000 -> 300000 (1500 steps/frame)
Parent rule: Highest UCB node
Observation: 1500 steps/frame WORKS! Matches Block 5 @100 frames R²=0.9998. F needs more steps at higher frame counts.
Next: parent=122

## Iter 123: excellent (omega_f=15 tested)
Node: id=123, parent=122
Mode/Strategy: success-exploit/exploit
Config: lr_NNR_f=3E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=15.0, batch_size=1
Metrics: final_r2=0.9996, final_mse=1.72E-4, slope=0.9996, training_time=15.7min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 20.0 -> 15.0
Parent rule: Highest UCB node (122)
Observation: omega_f=15 slightly worse than omega_f=20 (0.9996 vs 0.9998). Confirms omega_f=20 still optimal at 200 frames.
Next: parent=123

## Iter 124: excellent (omega_f=25 tested - plateau confirmed)
Node: id=124, parent=123
Mode/Strategy: success-exploit/exploit
Config: lr_NNR_f=3E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.9997, final_mse=1.22E-4, slope=0.9996, training_time=15.4min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 15.0 -> 25.0
Parent rule: Highest UCB node (123)
Observation: omega_f=25 R²=0.9997 ≈ omega_f=20 (0.9998) > omega_f=15 (0.9996). Confirms 15-25 plateau.
Next: parent=124

---

### Emerging Observations

- Block 11: F field data scaling test with 200 frames
- Iter 121: 1000 steps/frame insufficient for 200 frames (R²=0.935)
- Iter 122: 1500 steps/frame WORKS! R²=0.9998 achieved (matches 100 frames)
- Iter 123: omega_f=15 tested, R²=0.9996 - slightly worse than omega_f=20
- Iter 124: omega_f=25 tested, R²=0.9997 ≈ omega_f=20. **PLATEAU CONFIRMED**.
- **INSIGHT**: F field steps/frame requirement scales with n_frames: 1000@100frames → 1500@200frames
- **OMEGA_F MAP at 200 frames**: omega_f=15(0.9996) < omega_f=25(0.9997) ≈ omega_f=20(0.9998) - PLATEAU at 15-25
- **DATA SCALING FINDING**: F at 200 frames shows NO diminishing returns (same R²=0.9998 as 100 frames)
- **3 consecutive R² > 0.95** → Trigger **failure-probe** to find boundaries
- Next: Test extreme parameters - omega_f=5 (low boundary) or lr=1E-4 (high lr boundary)

