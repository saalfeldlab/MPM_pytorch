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
| 11    | siren_txy | F    | 200      | 0.9999  | 1.000      | 3E-5             | 256                | 4                  | 20.0            | 300000              | 18.0                | Data scaling SUCCESS (NO diminishing returns). All boundaries mapped. Speed Pareto: 256×2 (R²=0.994, 9.5min) |
| 12    | siren_txy | C    | 200      | 0.994   | 0.981      | 3E-5             | 384                | 3                  | 25.0            | 500000              | 28.9                | Data scaling HURTS C (0.994@200f < 0.996@100f). omega_f=25 optimal. Depth-sensitive like Jp. |
| 13    | siren_txy | S    | 48       | 0.801   | 0.846      | 2E-5             | 1280               | 4                  | 50.0            | 150000              | 58.8                | NEW RECORD. Capacity ceiling at 1280. S unique: omega_f=50, lr=2E-5, depth=4. |
| 14    | siren_txy | F    | 500      | 0.9998  | 1.000      | 3E-5             | 256                | 4                  | 20.0            | 400000              | 20.5                | DATA SCALING CONFIRMED: F scales to 500f with NO diminishing returns. Speed Pareto: 400 steps/frame (10min, R²=0.992) |
| 15    | siren_txy | Jp   | 500      | 0.997   | 0.968      | 4E-5             | 384                | 3                  | 15.0            | 400000              | 23.9                | Jp scales to 500f (0.997>0.989). OVERFITTING FIX: 800 steps/frame (not 2000). omega_f=15 optimal. |
| 16    | siren_txy | C    | 500      | 0.989   | 0.973      | 3E-5             | 640                | 3                  | 20.0            | 500000              | 51.7                | DATA SCALING PENALTY MITIGATED: 0.989 via capacity increase (640>384). C needs 1000 steps/frame. Depth=3 strict. |

### Established Principles
1. **omega_f sensitivity**: Field-specific optimal frequencies - Jp→15@500f/20-25@200f/35@48f, F→20, C→20@500f/25@200f/30@48f, S→50. More frames → lower optimal omega_f.
2. **lr optimal zone**: F→3E-5-1E-4 (widest), Jp→4E-5 (500 frames), C→2.5E-5-3E-5, S→2E-5 (strictest)
3. **hidden_dim field-dependent**: F needs 256, Jp needs 384, C needs 640@500f (capacity scales with frames), S needs 1280. Capacity ceiling: S@1280 (1536 FAILS), Jp@384 (both 256 AND 512 DEGRADE), C@640 (768 regresses).
4. **n_layers optimal**: F→2-5 layers (ALL work at 200f, but 4 REQUIRED at 500f), Jp→STRICTLY 3 (both 2 and 4 degrade), C→STRICTLY 3 (both 2 and 4 degrade), S→4 layers.
5. **Depth > width for F**: Adding depth more efficient than width for F field. Other fields prefer width.
6. **total_steps**: Field-dependent steps/frame ratio. F~800 (at 500f), Jp~800 at 500f, C~1000 at 500f, S~3000+ steps/frame.
7. **batch_size**: Use batch_size=1 to avoid training time explosion
8. **Field difficulty ranking**: F (R²=0.9999) > Jp (R²=0.997) > C (R²=0.989) >> S (R²=0.801). Stress tensor fundamentally harder.
9. **Local optimum detection**: When 5+ mutations from best node all regress, config-level is exhausted
10. **DATA SCALING CATEGORIZATION**: F benefits (no diminishing returns to 500f), Jp benefits (with overfitting fix at 500f), C HURTS (penalty mitigated by capacity increase to 0.989), S HURTS (fails entirely).
11. **LR-omega_f interaction**: LR tolerance widens at lower omega_f values
12. **Pareto-optimal configs**: F→256×4 (accuracy) or 256×2/128×4 (speed), Jp→384×3, C→640×3@500f, S→1280×4
13. **STOCHASTIC VARIANCE**: S field has EXTREME variance (R² 0.08-0.80). C has moderate variance (~0.007). F most stable.
14. **OVERFITTING LIMIT**: More training steps beyond optimal causes overfitting (Jp@200 frames: 400k OK, 500k regresses; C@500f: 500k OK, 600k regresses)
15. **Jp and C DEPTH SENSITIVE**: Both require exactly n_layers=3. F tolerates 2-5 layers.
16. **SPEED VS ACCURACY PARETO**: F@500: 128×4 (17.3min, R²=0.9966) vs 256×4 (20.5min, R²=0.9998). C@500: 640×3 (51.7min, R²=0.989).
17. **EFFICIENCY IMPROVES WITH DATA (F only)**: F needs only 800 steps/frame at 500 frames (vs 1500 at 200 frames).
18. **DEPTH MAP COMPLETE FOR ALL FIELDS**: F→[2-5] all viable, Jp→3 only, C→3 only, S→4 preferred. C and Jp share strict depth=3 requirement.

### Open Questions
1. Can S field be stabilized with code modification? (Loss scaling, gradient clipping, normalization)
2. Can S field achieve more consistent results with SMALLER network to reduce variance?
3. ~~What is the scaling behavior at 500+ frames?~~ **ANSWERED Block 14: F scales excellently, no diminishing returns**
4. Can different INR architectures (ngp, siren_t) match siren_txy performance?

---

## Previous Block Summary (Block 16)

**Field**: C, **INR Type**: siren_txy, **n_frames**: 500
**Best achieved**: R²=0.989 (Nodes 187, 191), hidden_dim=640, n_layers=3, omega_f=20, lr=2.5-3E-5, total_steps=500k
**Key findings**:
1. C DATA SCALING PENALTY MITIGATED: 0.989@500f via capacity increase (640>384). Still worse than 100f (0.996) but gap narrowed.
2. CAPACITY CEILING: 384(0.980)→512(0.987)→640(0.989)→768(0.979 REGRESS). Peak at 640.
3. omega_f=20 optimal (pattern confirmed: 30@48f→25@200f→20@500f)
4. DEPTH SENSITIVE: n_layers=2(0.953) << n_layers=3(0.989) >> n_layers=4(0.983). C shares Jp's strict depth=3.
5. 500-FRAME RANKING: F(0.9998) > Jp(0.997) > C(0.989) >> S(untested). C needs most capacity/time for worst accuracy among viable fields.

---

## Current Block (Block 17)

### Block Info
Field: field_name=S, inr_type=siren_txy
Iterations: 193-204
n_training_frames: 48

### Hypothesis
Testing S field with CODE MODIFICATION approach to address stochastic variance and capacity ceiling.
- S field has hit config-level ceiling (R²=0.801 max, extreme variance 0.08-0.80)
- Block 13 found optimal config: 1280×4, omega_f=50, lr=2E-5, steps=150k
- NEW APPROACH: Code modification to test if gradient clipping or LayerNorm can stabilize S field
- Primary goal: Reduce stochastic variance, not just increase R² ceiling
- Secondary goal: Determine if S field fundamentally requires different architecture or is just high-variance

### Iterations This Block

## Iter 193: moderate (S FIELD BASELINE - VARIANCE CONFIRMED)
Node: id=193, parent=root
Mode/Strategy: explore/root
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.751, final_mse=3.69E-08, slope=0.804, training_time=58.8min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: Block boundary - same config as Block 13 optimal
Parent rule: Root node (new block start)
Observation: R²=0.751 below Block 13 record (0.801). VARIANCE CONFIRMED. CODE MODIFICATION: Adding gradient clipping.
Next: parent=193 (gradient clipping code mod)

## Iter 194: moderate (GRADIENT CLIPPING SHOWS IMPROVEMENT)
Node: id=194, parent=193
Mode/Strategy: code-modification test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: Added gradient clipping (max_norm=1.0) after loss.backward()
Metrics: final_r2=0.785, final_mse=3.19E-08, slope=0.838, training_time=59.0min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: [code] gradient clipping max_norm=1.0
Parent rule: Node 193 (test code modification)
Observation: R²=0.785 vs 0.751 (parent) = +0.034 improvement. With clipping: range 0.034 (2 runs). Without: range 0.717 (5 runs). PROMISING variance reduction.
Next: parent=194 (robustness test - re-run same config)

## Iter 195: moderate (GRADIENT CLIPPING VARIANCE REDUCTION CONFIRMED)
Node: id=195, parent=194
Mode/Strategy: robustness-test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: (same as iter 194) gradient clipping max_norm=1.0 active
Metrics: final_r2=0.787, final_mse=3.14E-08, slope=0.837, training_time=59.1min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: Robustness test (same config)
Parent rule: UCB=2.011 (Node 195 highest) - robustness verification
Observation: VARIANCE REDUCTION CONFIRMED: With clipping: [0.785, 0.787], range=0.002. Without: range=0.717. 99.7% variance reduction! But mean R²~0.786 < Block 13 best (0.801). Clipping may be too aggressive.
Next: parent=195 (modify code: relax max_norm from 1.0 to 5.0)

## Iter 196: poor (MAX_NORM=5.0 CATASTROPHIC FAILURE)
Node: id=196, parent=195
Mode/Strategy: code-modification test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: Changed gradient clipping max_norm from 1.0 to 5.0
Metrics: final_r2=0.128, final_mse=1.33E-07, slope=0.203, training_time=59.2min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: [code] max_norm: 1.0 -> 5.0
Parent rule: UCB=1.730 (Node 195 highest)
Observation: CATASTROPHIC FAILURE. max_norm=5.0 TOO PERMISSIVE. S field requires STRICT clipping (max_norm=1.0). REVERTING to max_norm=1.0, will test max_norm=2.0 as smaller increment.
Next: parent=195 (revert to max_norm=1.0, test max_norm=2.0)

## Iter 197: poor (MAX_NORM=2.0 ALSO FAILS - SHARP THRESHOLD CONFIRMED)
Node: id=197, parent=195
Mode/Strategy: code-modification test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: Changed gradient clipping max_norm from 5.0 to 2.0
Metrics: final_r2=0.118, final_mse=1.34E-07, slope=0.188, training_time=59.2min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: [code] max_norm: 5.0 -> 2.0
Parent rule: UCB=1.709 (Node 196 highest but failed)
Observation: ALSO FAILS. max_norm=2.0 equally catastrophic (R²=0.118). SHARP THRESHOLD: max_norm must be ≤1.0. Clipping map: 1.0→0.786, 2.0→0.118, 5.0→0.128. Testing max_norm=1.5 next.
Next: parent=195 (revert to max_norm=1.0, test max_norm=1.5)

## Iter 198: poor (MAX_NORM=1.5 FAILS - THRESHOLD AT EXACTLY 1.0)
Node: id=198, parent=195
Mode/Strategy: code-modification test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: Changed gradient clipping max_norm from 2.0 to 1.5
Metrics: final_r2=0.075, final_mse=1.46E-07, slope=0.152, training_time=59.2min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: [code] max_norm: 2.0 -> 1.5
Parent rule: UCB=1.807 (Node 198 highest in UCB list)
Observation: ALSO FAILS (R²=0.075, worst yet). THRESHOLD EXACTLY AT 1.0. Gradient clipping map COMPLETE: 0.5→?, 1.0→0.786, 1.5→0.075, 2.0→0.118, 5.0→0.128. Testing max_norm=0.5 next to see if tighter clipping helps.
Next: parent=195 (test max_norm=0.5)

## Iter 199: moderate (MAX_NORM=0.5 NEW S FIELD RECORD!)
Node: id=199, parent=195
Mode/Strategy: code-modification test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: Changed gradient clipping max_norm from 1.5 to 0.5
Metrics: final_r2=0.828, final_mse=2.56E-08, slope=0.893, training_time=59.2min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: [code] max_norm: 1.5 -> 0.5
Parent rule: UCB=2.699 (Node 199 highest)
Observation: NEW S FIELD RECORD! R²=0.828 (+0.027 vs Block 13). TIGHTER CLIPPING IMPROVES R². Clipping map: 0.5→0.828 (NEW BEST), 1.0→0.786, 1.5→0.075, 2.0→0.118, 5.0→0.128.
Next: parent=199 (robustness test to verify stability, then try max_norm=0.25)

## Iter 200: moderate (MAX_NORM=0.25 REGRESSES - OPTIMAL AT 0.5)
Node: id=200, parent=199
Mode/Strategy: code-modification test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: Changed gradient clipping max_norm from 0.5 to 0.25
Metrics: final_r2=0.810, final_mse=2.81E-08, slope=0.861, training_time=59.2min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: [code] max_norm: 0.5 -> 0.25
Parent rule: UCB=2.809 (Node 200 highest)
Observation: REGRESSION. R²=0.810 < 0.828. max_norm=0.25 TOO TIGHT. OPTIMAL FOUND: max_norm=0.5. Clipping map COMPLETE: 0.25→0.810, 0.5→0.828 (OPTIMAL), 1.0→0.786, 1.5→0.075, 2.0→0.118, 5.0→0.128.
Next: parent=199 (revert to max_norm=0.5, run robustness test)

## Iter 201: poor (MAX_NORM=0.5 ROBUSTNESS FAILS - VARIANCE STILL HIGH)
Node: id=201, parent=199
Mode/Strategy: robustness-test
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
CODE MODIFICATION: (same as iter 199) gradient clipping max_norm=0.5 active
Metrics: final_r2=0.181, final_mse=1.26E-07, slope=0.276, training_time=59.3min
Field: field_name=S, inr_type=siren_txy, n_frames=48
Mutation: Robustness test (same config as iter 199)
Parent rule: UCB=2.931 (Node 200 highest), testing node 199's optimal max_norm=0.5
Observation: CATASTROPHIC VARIANCE. max_norm=0.5 runs: [0.828, 0.181], range=0.647. GRADIENT CLIPPING ALONE DOES NOT SOLVE S FIELD VARIANCE! Need LayerNorm.
Next: parent=199 (add LayerNorm to Siren network in addition to gradient clipping)

---

### Emerging Observations

1. **Iteration 193**: S field baseline with optimal config (1280×4, omega=50, lr=2E-5) yields R²=0.751, below the 0.801 record. Confirms stochastic variance.
2. **Iteration 194**: Gradient clipping (max_norm=1.0) improves R² from 0.751 to 0.785 (+0.034). Variance with clipping << variance without.
3. **Iteration 195 - VARIANCE REDUCTION CONFIRMED (PREMATURE)**: With clipping@1.0: [0.785, 0.787], range=0.002 (2 samples). Appeared to reduce variance by 99.7%.
4. **Iteration 196-198 - CLIPPING MAP UPPER BOUNDS**: max_norm > 1.0 ALL FAIL catastrophically: 1.5→0.075, 2.0→0.118, 5.0→0.128. THRESHOLD EXACTLY AT 1.0.
5. **Iteration 199 - NEW RECORD!**: max_norm=0.5 achieves R²=0.828 (+0.027 vs previous best 0.801). **TIGHTER CLIPPING IMPROVES R²!**
6. **Iteration 200 - max_norm=0.25 REGRESSES**: R²=0.810 < 0.828. **OPTIMAL FOUND AT max_norm=0.5**. Too tight also hurts.
7. **Iteration 201 - CRITICAL: VARIANCE STILL HIGH!**: max_norm=0.5 robustness test FAILS: [0.828, 0.181], range=0.647. **GRADIENT CLIPPING ALONE DOES NOT SOLVE S FIELD VARIANCE!** The variance at max_norm=0.5 (0.647) is similar to without clipping (0.717). CLIPPING ONLY IMPROVED PEAK, NOT STABILITY.
8. **GRADIENT CLIPPING MAP (COMPLETE)**: 0.25→0.810, 0.5→[0.828,0.181], 1.0→[0.785,0.787], 1.5→0.075, 2.0→0.118, 5.0→0.128
9. **KEY INSIGHT - VARIANCE PERSISTS**: max_norm=1.0 appears MOST STABLE (range=0.002) but lower R² (0.786). max_norm=0.5 has HIGHER PEAK (0.828) but HIGHER VARIANCE (0.647). S field has STABILITY-PERFORMANCE TRADEOFF.
10. **NEXT STEP**: Try adding LayerNorm to Siren network in addition to gradient clipping. LayerNorm may stabilize activations and reduce initialization-dependent variance. Keep max_norm=0.5 (highest peak) and add LayerNorm between hidden layers.


