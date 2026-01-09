# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1     | siren_txy | Jp   | 48       | 0.968   | 0.901      | 2.5E-5           | 512                | 3                  | 35.0            | 100000              | 8.8                 | omega_f=35 optimal, lr zone 2E-5→3E-5 |
| 2     | siren_txy | F    | 48       | 0.9995  | 0.9995     | 3E-5             | 256                | 4                  | 25.0            | 100000              | 6.3                 | omega_f=25, F easier than Jp, 256 sufficient |
| 3     | siren_txy | S    | 48       | 0.618   | 0.629      | 2E-5             | 512                | 4                  | 50.0            | 150000              | 16.7                | S HARD field, R²=0.618 ceiling, omega_f=50 sharp peak |
| 4     | siren_txy | C    | 48       | 0.993   | 0.981      | 3E-5             | 384                | 3                  | 30.0            | 100000              | 7.1                 | omega_f=30, C like F (easy), 384 optimal |

### Established Principles
1. **omega_f sensitivity**: Field-specific optimal frequencies - Jp→35, F→25, C→30, S→50.
2. **lr optimal zone**: lr_NNR_f in [2E-5, 3E-5]; field-specific (F/C→3E-5, S/Jp→2E-5)
3. **hidden_dim field-dependent**: Jp/S need 512, F needs 256, C needs 384. Capacity ceiling exists per field.
4. **n_layers optimal**: 3-4 layers optimal for siren_txy; 2 insufficient, 5 degrades
5. **Depth > width**: Adding depth more efficient than width for SIREN
6. **total_steps**: 100k for 48 frames (easy fields), 150k for S (hard field). Excess steps can hurt (overfitting).
7. **batch_size**: Use batch_size=1 to avoid training time explosion
8. **Field difficulty ranking**: F (R²=0.9995) > C (R²=0.993) >> Jp (R²=0.968) >> S (R²=0.618). Stress tensor fundamentally harder.
9. **Local optimum detection**: When 5+ mutations from best node all regress, config-level is exhausted

### Open Questions
1. Does increasing n_training_frames (48→100) improve R² for easy fields (F, C)?
2. Does S field require code modification (loss scaling/normalization) to exceed R²=0.618?
3. Will n_frames=100 require proportionally more total_steps?

---

## Previous Block Summary (Block 4)

**Field**: C (APIC matrix), **INR Type**: siren_txy, **n_frames**: 48
Best config: lr=3E-5, hidden_dim=384, n_layers=3, omega_f=30.0, 100k steps → R²=0.993, slope=0.981, 7.1min
Key findings: C field SUCCESS (all 12 iters R²>0.97). omega_f=30 optimal. n_layers=3, hidden_dim=384 sweet spots. Overfitting detected (150k < 100k steps).
Branching rate: 33%, Improvement rate: 50%.

---

## Current Block (Block 5)

### Block Info
Field: field_name=F, inr_type=siren_txy
Iterations: 49-60
n_training_frames: 100 (INCREASED from 48)

### Hypothesis
Testing data scaling: F field achieved R²=0.9995 at 48 frames. With 100 frames (2× data), expect:
- R² may slightly decrease initially due to harder task (more timesteps to fit)
- total_steps=150k should be sufficient (1500 steps/frame baseline from Block 2 knowledge)
- If R² decreases significantly, will test 200k steps
- Using F-optimal config from Block 2: lr=3E-5, hidden_dim=256, n_layers=4, omega_f=25

### Iterations This Block

## Iter 49: excellent
Node: id=49, parent=root
Mode/Strategy: exploit (first iteration of block)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.9998, final_mse=8.6E-5, total_params=265220, slope=0.9999, training_time=8.5min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: n_frames: 48 -> 100 (2× data scaling test), total_steps: 100k -> 150k
Parent rule: Block start - using Block 2 optimal F config with scaled steps for more frames
Observation: EXCELLENT! R²=0.9998 with 100 frames EXCEEDS Block 2's R²=0.9995 with 48 frames. More data HELPS, not hurts.
Next: parent=49 (only node, UCB=1.707)

## Iter 50: excellent
Node: id=50, parent=49
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.9987, final_mse=6.0E-4, total_params=265220, slope=0.9986, training_time=6.4min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: total_steps: 150000 -> 100000
Parent rule: Highest UCB node (1.707), test if fewer steps maintain quality
Observation: SUCCESS! 100k steps achieves R²=0.9987 (vs 150k→0.9998). Slight decrease but still excellent, 25% faster.
Next: parent=50 (UCB=1.999, testing 80k steps for speed-accuracy tradeoff)

## Iter 51: excellent
Node: id=51, parent=50
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=80000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.983, final_mse=8.2E-3, total_params=265220, slope=0.981, training_time=5.2min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: total_steps: 100000 -> 80000
Parent rule: Highest UCB node (1.999), testing minimum steps for speed
Observation: 80k steps (800/frame) hits lower bound - R²=0.983 drops below 0.99. 100k steps (1000/frame) is minimum for R²>0.99.
Next: parent=51 (UCB=2.208, explore architecture variation)

### Emerging Observations

1. **Data scaling SUCCESS**: F field R²=0.9998 with 100 frames > R²=0.9995 with 48 frames. More training data HELPS.
2. **Steps/frame ratio**: 150k→1500/frame (R²=0.9998), 100k→1000/frame (R²=0.9987), 80k→800/frame (R²=0.983). Sweet spot: 1000 steps/frame.
3. **Training time scaling**: 8.5min (150k) → 6.4min (100k) → 5.2min (80k). Linear with steps.
4. **Hypothesis validated**: F-optimal config (256×4, omega_f=25, lr=3E-5) generalizes to 100 frames.
5. **Step minimum found**: 80k steps (R²=0.983) below 0.99 threshold. 100k steps is minimum for R²>0.99 with 100 frames.
6. **Next direction**: Testing architecture variation (hidden_dim or n_layers) from 80k/100k configs.

