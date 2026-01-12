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

### Established Principles
1. **omega_f sensitivity**: Field-specific optimal frequencies - Jp→30-35, F→15-25, C→30, S→50. More frames → lower optimal omega_f (except S).
2. **lr optimal zone**: lr_NNR_f in [2E-5, 4E-5]; F→3E-5-4E-5 at low omega_f, Jp/C→3E-5, S→2E-5 (strict)
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

### Open Questions
1. Can C field improve beyond R²=0.993 with 100 frames? (Testing in Block 8)
2. Can S field improve beyond R²=0.708 with even more capacity (1024×4)?
3. Is there a code modification that could help S field? (Loss scaling, normalization)

---

## Previous Block Summary (Block 7)

**Field**: S (stress tensor), **INR Type**: siren_txy, **n_frames**: 48→100
Best config: lr=2E-5, hidden_dim=768, n_layers=4, omega_f=50, 250k steps, 48 frames → R²=0.708, slope=0.735
Key findings: Data scaling FAILS for S (100 frames worse than 48). Capacity scaling WORKS (768>>512, +0.090 R²). NEW S RECORD R²=0.708.
Block stats: 4/12 moderate (33%), 8/12 poor (67%). Branching rate: 42%.

---

## Current Block (Block 8)

### Block Info
Field: field_name=C, inr_type=siren_txy
Iterations: 85-96
n_training_frames: 100 (testing data scaling on C field)

### Hypothesis
Testing if data scaling helps C field (APIC matrix). Block 4 found R²=0.993 at 48 frames with 384×3.
- Using C-optimal config from Block 4: lr=3E-5, hidden_dim=384, n_layers=3, omega_f=30
- total_steps=100k initially (1000 steps/frame like F)
- Expect: If C behaves like F (easy field), data scaling should help. If like S, it will regress.
- Prediction: C is similar to F, so expect R² improvement toward 0.995+

### Iterations This Block

## Iter 85: good
Node: id=85, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.972, final_mse=5.34E+00, slope=0.965, training_time=21.9min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: New block - C field with 100 frames (data scaling test)
Observation: SURPRISING REGRESSION! 100 frames (R²=0.972) < 48 frames (R²=0.993). Data scaling FAILED for C.
Next: parent=85, test total_steps=200k (2000 steps/frame like Jp)

## Iter 86: excellent
Node: id=86, parent=85
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.996, final_mse=7.33E-01, slope=0.990, training_time=43.4min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 100000 -> 200000
Observation: MAJOR RECOVERY! 200k steps (R²=0.996) >> 100k steps (R²=0.972). C data scaling WORKS with 2000 steps/frame. EXCEEDS 48-frame best.
Next: parent=86, test omega_f=25

## Iter 87: excellent
Node: id=87, parent=86
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.995, final_mse=9.76E-01, slope=0.991, training_time=43.4min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 30.0 -> 25.0
Observation: omega_f=25 slightly worse than omega_f=30 (R²=0.995 vs 0.996, -0.001). C optimal omega_f stays ~30 even with 100 frames.
Next: parent=86, test hidden_dim=512

## Iter 88: excellent
Node: id=88, parent=86
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.991, final_mse=1.67E+00, slope=0.985, training_time=74.7min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 384 -> 512
Observation: hidden_dim=512 REGRESSED from 384 (R²=0.991 vs 0.996, -0.005). Confirms 384 optimal for C. Over-capacity, 1.7× slower.
Next: parent=87, test omega_f=35

## Iter 89: excellent
Node: id=89, parent=87
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.996, final_mse=8.51E-01, slope=0.987, training_time=43.4min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 25.0 -> 35.0
Observation: omega_f=35 MATCHES omega_f=30 (both R²=0.996). C field omega_f range [25-35] is a plateau. omega_f=30 remains optimal (slightly better slope).
Next: parent=89, test lr=4E-5

## Iter 90: excellent
Node: id=90, parent=89
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.996, final_mse=7.28E-01, slope=0.991, training_time=43.4min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 3E-5 -> 4E-5, omega_f: 35.0 -> 30.0
Observation: lr=4E-5 MATCHES lr=3E-5 (both R²=0.996). C field has WIDE lr tolerance [3E-5, 4E-5] at omega_f=30. Confirms LR-omega_f interaction.
Next: parent=90, test n_layers=4

## Iter 91: excellent
Node: id=91, parent=90
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.992, final_mse=1.48E+00, slope=0.990, training_time=55.9min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: n_layers_nnr_f: 3 -> 4
Observation: n_layers=4 REGRESSED (R²=0.992 vs 0.996, -0.004). CONFIRMS C optimal depth=3 (like Jp). Extra layer 33% more params, 29% slower, worse R².
Next: parent=91 (UCB=2.863), test lr=5E-5 (lr upper boundary) with n_layers=3 revert

## Iter 92: excellent
Node: id=92, parent=91
Mode/Strategy: failure-probe (7 consecutive excellent)
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.996, final_mse=8.48E-01, slope=0.989, training_time=43.4min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 4E-5 -> 5E-5, n_layers_nnr_f: 4 -> 3 (revert)
Observation: lr=5E-5 STILL WORKS (R²=0.996). C field lr tolerance [3E-5, 5E-5] widest of all fields.
Next: parent=92 (UCB=2.995), test lr=6E-5 (extreme lr boundary probe)

## Iter 93: excellent
Node: id=93, parent=92
Mode/Strategy: failure-probe (8 consecutive excellent/good)
Config: lr_NNR_f=6E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.993, final_mse=1.42E+00, slope=0.987, training_time=43.4min
Field: field_name=C, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 5E-5 -> 6E-5
Observation: lr=6E-5 slight regression (R²=0.993 vs 0.996). LR UPPER BOUNDARY FOUND: C field lr tolerance [3E-5, 5E-5].
Next: parent=93 (UCB=3.114), test omega_f=40 (omega_f upper boundary probe) with lr=5E-5 revert

### Emerging Observations

- **C field data scaling WORKS with sufficient steps**: 200k steps (R²=0.996) > 48 frames best (R²=0.993). C needs 2000 steps/frame like Jp.
- **Field steps/frame requirements confirmed**: F~1000, Jp~2000, C~2000. S unclear (data scaling fails regardless).
- **C omega_f plateau [25-35]**: omega_f=25/30/35 all yield R²=0.995-0.996 (plateau). 30 marginally best for slope (0.990-0.991).
- **hidden_dim=384 CONFIRMED optimal for C**: 512 causes regression (0.991 vs 0.996) and 1.7× slower. Consistent with Block 4.
- **C field lr tolerance BOUNDED**: lr=3E-5, 4E-5, 5E-5 yield R²=0.996. lr=6E-5 regresses to R²=0.993. C field lr range: [3E-5, 5E-5].
- **n_layers=3 CONFIRMED optimal for C**: 4 layers regresses (R²=0.992 vs 0.996) with 33% more params & 29% slower. Same pattern as Jp (depth-sensitive).
- **Current block progress**: 8 excellent (iter 86-93), 1 good. 89% excellent rate. Best=R²=0.996 at nodes 86, 89, 90, 92.
- **LR UPPER BOUNDARY FOUND**: lr=5E-5 works, lr=6E-5 regresses. Next: test omega_f=40 (upper boundary probe).

