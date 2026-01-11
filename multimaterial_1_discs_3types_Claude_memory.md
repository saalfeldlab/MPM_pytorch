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

### Established Principles
1. **omega_f sensitivity**: Field-specific optimal frequencies - Jp→30-35, F→15-25, C→30, S→50. More frames → lower optimal omega_f.
2. **lr optimal zone**: lr_NNR_f in [2E-5, 4E-5]; F→3E-5-4E-5 at low omega_f, Jp/C→3E-5, S→2E-5
3. **hidden_dim field-dependent**: F needs 256, Jp/C need 384, S needs 512. Capacity ceiling exists per field.
4. **n_layers optimal**: F→4 layers, Jp/C/S→3 layers. Jp is depth-sensitive (4 layers degrades severely).
5. **Depth > width for F**: Adding depth more efficient than width for F field. Other fields prefer width.
6. **total_steps**: Field-dependent steps/frame ratio. F~1000, Jp~2000 steps/frame for R²>0.98.
7. **batch_size**: Use batch_size=1 to avoid training time explosion
8. **Field difficulty ranking**: F (R²=0.9998) > C (R²=0.993) >> Jp (R²=0.982) >> S (R²=0.618). Stress tensor fundamentally harder.
9. **Local optimum detection**: When 5+ mutations from best node all regress, config-level is exhausted
10. **Data scaling benefit CONFIRMED**: Both F and Jp improve with more frames. F: 48→100 frames = R²+0.0003, Jp: 48→100 = R²+0.014
11. **LR-omega_f interaction**: LR tolerance widens at lower omega_f values
12. **256×4 Pareto-optimal for F, 384×3 Pareto-optimal for Jp/C**

### Open Questions
1. Does S field benefit from data scaling (48→100 frames), or is it fundamentally limited?
2. Can C field improve beyond R²=0.993 with 100 frames?
3. What is the optimal omega_f for S field at 100 frames?

---

## Previous Block Summary (Block 6)

**Field**: Jp (plastic deformation), **INR Type**: siren_txy, **n_frames**: 100
Best config: lr=3E-5, hidden_dim=384, n_layers=3, omega_f=30, 200k steps → R²=0.982, slope=0.938
Key findings: Data scaling SUCCESS (+0.014 R² vs 48 frames). hidden_dim=384 optimal (>512>256). omega_f shifted 35→30 with more data. n_layers=3 strictly required (4 layers regresses to R²=0.838).
Block stats: 8/12 excellent (67%), branching rate 17%.

---

## Current Block (Block 7)

### Block Info
Field: field_name=S, inr_type=siren_txy
Iterations: 73-84
n_training_frames: 100 (testing data scaling on HARD field)

### Hypothesis
Testing if data scaling helps the HARD field S (stress tensor). Block 3 found R²=0.618 ceiling at 48 frames.
- Using S-optimal config from Block 3: lr=2E-5, hidden_dim=512, n_layers=4, omega_f=50
- total_steps=200k (2000 steps/frame, following Jp pattern)
- If S field is fundamentally limited (not data-limited), R² will plateau near 0.618 regardless of frames.
- If data-limited like F/Jp, expect R² improvement toward 0.70+

### Iterations This Block

## Iter 73: poor
Node: id=73, parent=root
Mode/Strategy: exploit (block start)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.517, final_mse=7.09E-8, total_params=1054724, slope=0.559, training_time=116.3min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: n_training_frames: 48 -> 100 (data scaling test)
Parent rule: root (new block)
Observation: S field WORSE than Block 3 ceiling (R²=0.517 vs 0.618). Training time EXPLOSION (116 min). Data scaling NOT immediately helping. CRITICAL: Need to reduce training time first.
Next: parent=73 (reduce steps for faster iteration)

## Iter 74: poor
Node: id=74, parent=73
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.456, final_mse=8.12E-8, total_params=1054724, slope=0.522, training_time=55.7min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 200000 -> 100000
Parent rule: Highest UCB (node 73, UCB=1.184)
Observation: Reducing steps HURT S field (R²=0.517→0.456). S field at 100 frames now WORSE than Block 3 ceiling. Training time acceptable (55.7min). Try omega_f reduction (following data scaling→lower freq pattern).
Next: parent=74 (omega_f reduction)

## Iter 75: poor
Node: id=75, parent=74
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=45.0, batch_size=1
Metrics: final_r2=0.359, final_mse=1.05E-7, total_params=1054724, slope=0.440, training_time=55.6min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 50.0 -> 45.0
Parent rule: Highest UCB (node 75, UCB=1.583)
Observation: omega_f reduction CATASTROPHIC (R²=0.456→0.359, -0.097). CONFIRMS omega_f=50 is SHARP PEAK for S. Data scaling does NOT shift S's optimal omega_f (unlike F/Jp). Backtracking.
Next: parent=73 (backtrack, try hidden_dim reduction)

## Iter 76: poor
Node: id=76, parent=73
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.487, final_mse=7.47E-8, total_params=594436, slope=0.495, training_time=64.9min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 512 -> 384
Parent rule: Backtrack to node 73 (best R²=0.517)
Observation: hidden_dim reduction WORSE (R²=0.517→0.487, -0.030). S needs higher capacity. 4/4 iterations ALL worse than Block 3 ceiling (R²=0.618). DATA SCALING FAILS for S field.
Next: parent=76 (try n_layers reduction 4→3)

## Iter 77: poor
Node: id=77, parent=76
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=50.0, batch_size=1
Metrics: final_r2=0.443, final_mse=8.52E-8, total_params=446596, slope=0.493, training_time=52.8min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: n_layers_nnr_f: 4 -> 3
Parent rule: Highest UCB (node 77, UCB=2.024)
Observation: n_layers reduction HURT (R²=0.487→0.443, -0.044). S field prefers 4 layers. 5/5 iterations at 100 frames ALL worse than Block 3 ceiling (R²=0.618). DATA SCALING FUNDAMENTALLY FAILS for S field.
Next: parent=73 (backtrack to best R²=0.517, try increased capacity 512→768)

## Iter 78: poor
Node: id=78, parent=73
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=768, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.590, final_mse=5.99E-8, total_params=2368516, slope=0.612, training_time=205.1min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 512 -> 768
Parent rule: Backtrack to node 73 (best R²=0.517, 512×4), try increased capacity
Observation: Capacity increase HELPED (R²=0.517→0.590, +0.073) but STILL below Block 3 ceiling (R²=0.618). Training time EXPLOSION (205.1min). S field benefits from capacity but 100 frames STILL worse than 48 frames.
Next: parent=78 (try LR increase 2E-5→3E-5)

### Emerging Observations

1. **S field data scaling FAILS**: ALL 6 iterations at 100 frames (R²=0.517-0.590) WORSE than Block 3 ceiling R²=0.618 at 48 frames.
2. **S field is NOT data-limited**: Unlike F/Jp, more data HURTS S field. This is a fundamental representation problem.
3. **omega_f=50 RIGID**: Does NOT shift lower with more data (unlike F/Jp).
4. **Capacity HELPS but not enough**: 768>512>384 for hidden_dim. n_layers=4 REQUIRED.
5. **Reducing steps hurts**: 200k→100k hurt R² by 0.061.
6. **Training time**: 768×4 = 205min (unacceptable). Need to find efficiency.
7. **CRITICAL FINDING**: 768×4 at 100 frames (R²=0.590) < 512×4 at 48 frames (R²=0.618). Data scaling actively hurts S field.
8. **Next strategy**: Try LR increase 2E-5→3E-5 on node 78. If fails, consider returning to 48 frames or code modification.

