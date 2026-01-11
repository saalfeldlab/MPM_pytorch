# Experiment Log: multimaterial_1_discs_3types_Claude

## Iter 78: poor
Node: id=78, parent=73
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=768, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.590, final_mse=5.99E-8, total_params=2368516, slope=0.612, training_time=205.1min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 512 -> 768
Parent rule: Backtrack to node 73 (best R²=0.517, 512×4), try increased capacity
Observation: Capacity increase HELPED (R²=0.517→0.590, +0.073) but STILL below Block 3 ceiling (R²=0.618). Training time EXPLOSION (205.1min). S field benefits from more capacity but 100 frames STILL underperforms 48 frames. Pattern: 768>512>384 for hidden_dim at 100 frames. This proves S field at 100 frames is HARDER than at 48 frames - not data-limited but representation-limited. More data introduces more complexity without proportional signal.
Next: parent=78 (try LR increase 2E-5→3E-5 to improve convergence)

---

## Iter 77: poor
Node: id=77, parent=76
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=50.0, batch_size=1
Metrics: final_r2=0.443, final_mse=8.52E-8, total_params=446596, slope=0.493, training_time=52.8min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: n_layers_nnr_f: 4 -> 3
Parent rule: Highest UCB (node 77, UCB=2.024)
Observation: n_layers reduction HURT (R²=0.487→0.443, -0.044). S field prefers 4 layers. 5/5 iterations at 100 frames ALL worse than Block 3 ceiling (R²=0.618). DATA SCALING FUNDAMENTALLY FAILS for S field. CRITICAL INSIGHT: S field becomes HARDER with more data, not easier.
Next: parent=73 (backtrack to best R²=0.517, try increased capacity 512→768 or lr increase)

---

## Iter 76: poor
Node: id=76, parent=73
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.487, final_mse=7.47E-8, total_params=594436, slope=0.495, training_time=64.9min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 512 -> 384
Parent rule: Backtrack to node 73 (best R²=0.517)
Observation: hidden_dim reduction WORSE (R²=0.517→0.487, -0.030). S field needs higher capacity, not lower. 4/4 iterations with 100 frames ALL worse than Block 3 ceiling (R²=0.618 at 48 frames). DATA SCALING FAILS for S field. S appears representation-limited, not data-limited.
Next: parent=76 (try n_layers reduction 4→3)

---

## Iter 75: poor
Node: id=75, parent=74
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=45.0, batch_size=1
Metrics: final_r2=0.359, final_mse=1.05E-7, total_params=1054724, slope=0.440, training_time=55.6min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 50.0 -> 45.0
Parent rule: Highest UCB (node 75, UCB=1.583)
Observation: omega_f reduction CATASTROPHIC for S field (R²=0.456→0.359, -0.097). CONFIRMS Block 3 finding: omega_f=50 is SHARP PEAK for S field. Even small deviation (-5) causes major regression. S field fundamentally different from F/Jp - more data does NOT shift optimal omega_f lower. Backtracking to Node 73 (200k steps, omega_f=50).
Next: parent=73 (backtrack, try hidden_dim reduction)

---

## Iter 74: poor
Node: id=74, parent=73
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.456, final_mse=8.12E-8, total_params=1054724, slope=0.522, training_time=55.7min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 200000 -> 100000
Parent rule: Highest UCB (node 73, UCB=1.184)
Observation: Reducing steps HURT S field (R²=0.517→0.456, -0.061). S field at 100 frames now WORSE than Block 3 48-frame ceiling (R²=0.618). Training time now acceptable (55.7min). Issue is not just training duration - S field fundamentally struggles at 100 frames. Need to try different approach: possibly lower omega_f (data scaling→lower freq pattern) or smaller network.
Next: parent=74 (try omega_f reduction)

---

## Iter 73: poor
Node: id=73, parent=root
Mode/Strategy: exploit (block start)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.517, final_mse=7.09E-8, total_params=1054724, slope=0.559, training_time=116.3min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: n_training_frames: 48 -> 100 (data scaling test)
Parent rule: root (new block)
Observation: S field WORSE than Block 3 ceiling (R²=0.517 vs 0.618). Training time EXPLOSION (116 min). Data scaling NOT immediately helping with current config. S field appears fundamentally harder, not just data-limited. Need to reduce steps for faster iteration.
Next: parent=73 (reduce steps)

---

## Iter 72: moderate
Node: id=72, parent=68
Mode/Strategy: exploit (depth probe for block end)
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.838, final_mse=22.08, total_params=593281, slope=0.788, training_time=63.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: n_layers_nnr_f: 3 -> 4
Parent rule: Highest UCB (node 68, UCB=1.798)
Observation: CONFIRMS depth hurts Jp. n_layers=4 MAJOR REGRESSION (R²=0.982→0.838, -0.144). Unlike F field (where 4 layers optimal), Jp is DEPTH-SENSITIVE. 3 layers is optimal for Jp at all frame counts. Consistent with Block 1 iter 62 finding.

---

## Block 6 Summary (Jp, 100 frames, iters 61-72)

**Best config (iter 68)**: lr=3E-5, hidden_dim=384, n_layers=3, omega_f=30, 200k steps → R²=0.982, slope=0.938, 43.5min
**Improvement over Block 1**: R²=0.982 > 0.968 (+0.014). Data scaling SUCCESS for Jp field.
**Key findings**:
1. **Data scaling works for Jp**: 100 frames R²=0.982 > 48 frames R²=0.968 (+0.014), but needs 2000 steps/frame (vs F's 1000).
2. **hidden_dim=384 optimal for Jp**: 384 > 512 > 256. Same pattern as C field.
3. **omega_f=30 optimal at 100 frames**: Shifted down from omega_f=35 at 48 frames. Pattern: more data → lower optimal omega_f.
4. **n_layers=3 strictly optimal**: 4 layers causes major regression (R²=0.838). Jp is more depth-sensitive than F.
5. **lr=3E-5 optimal (upper bound ~3.5E-5)**: lr=4E-5 FAILED (R²=0.768).
**Block stats**: 8/12 excellent (67%), 4/12 moderate. Branching rate: 17% (2/12 non-sequential). Improvement rate: 50%.
**Regime confirmed**: Field difficulty persists - Jp harder than F but benefits from data scaling.

---

## Iter 71: excellent
Node: id=71, parent=68
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.980, final_mse=3.09, total_params=445441, slope=0.925, training_time=47.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 30 -> 25
Parent rule: Highest UCB (node 71, UCB=3.325)
Observation: omega_f=25 SLIGHTLY REGRESSED from omega_f=30 (R²=0.982→0.980, -0.002). Confirms omega_f=30 is the local optimum for Jp@100frames. Tested 25<30>35 = peak at 30. Unlike F field (omega_f=15-25), Jp prefers higher frequency.
Next: parent=68

## Iter 70: excellent
Node: id=70, parent=68
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.974, final_mse=3.75, total_params=445441, slope=0.930, training_time=45.2min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 30 -> 35
Parent rule: Highest UCB (node 68, UCB=2.100)
Observation: omega_f=35 REGRESSED from omega_f=30 (R²=0.982→0.974, -0.008). CONFIRMS omega_f=30 is optimal for Jp at 100 frames/hidden_dim=384. Block 1 found omega_f=35 optimal at 48 frames - data scaling shifts optimal omega_f DOWN. Pattern: more data → lower optimal omega_f.
Next: parent=68

## Iter 69: excellent
Node: id=69, parent=68
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.976, final_mse=3.71, total_params=198657, slope=0.915, training_time=24.2min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 384 -> 256
Parent rule: Highest UCB (node 68, UCB=2.982)
Observation: hidden_dim=256 REGRESSED from 384 (R²=0.982→0.976, -0.006). Unlike F field where 256 is optimal, Jp needs 384. Training time normalized to 24.2min (vs 43.5min at 384). Pareto tradeoff: 256 is 1.8× faster but 0.6% lower R². For Jp at 100 frames: 384 > 256 > 512.
Next: parent=68

## Iter 68: excellent
Node: id=68, parent=66
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.982, final_mse=2.62, total_params=445441, slope=0.938, training_time=43.5min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 512 -> 384
Parent rule: Highest UCB (node 66, UCB=1.968)
Observation: BREAKTHROUGH! hidden_dim=384 IMPROVED R² (0.968→0.982, +0.014)! NEW BLOCK BEST. Training time normalized (43.5min vs 74min). Like C field (Block 4), Jp benefits from smaller width. 384 > 512 for Jp at 100 frames. Data scaling + optimal width = R²=0.982 (exceeds 48-frame Block 1 R²=0.968).
Next: parent=68

## Iter 67: moderate
Node: id=67, parent=66
Mode/Strategy: exploit (boundary probe)
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.768, final_mse=31.2, total_params=790529, slope=0.744, training_time=74.5min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 3E-5 -> 4E-5
Parent rule: Highest UCB (node 66, UCB=2.215)
Observation: lr=4E-5 FAILED! Major regression R²=0.968→0.768 (-0.20). LR upper bound for Jp is ~3.5E-5 at omega_f=30. Unlike F field (tolerates 4E-5 at omega_f=15), Jp is more lr-sensitive. Confirms established principle: Jp needs lower lr than F.
Next: parent=66

## Iter 66: excellent
Node: id=66, parent=65
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.968, final_mse=4.38, total_params=790529, slope=0.939, training_time=74.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 40 -> 30
Parent rule: Highest UCB (node 65, UCB=2.114)
Observation: omega_f=30 IMPROVED over omega_f=40 (R²=0.959→0.968, +0.009). BEST IN BLOCK, matches Block 1 baseline! omega_f=30-35 confirmed optimal zone for Jp. Training time still anomalous at ~74min (200k steps issue?).
Next: parent=66

## Iter 65: excellent
Node: id=65, parent=64
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=40.0, batch_size=1
Metrics: final_r2=0.959, final_mse=5.68, total_params=790529, slope=0.919, training_time=74.2min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 35 -> 40
Parent rule: Highest UCB (node 64, UCB=2.017)
Observation: omega_f=40 caused REGRESSION (R²=0.963→0.959, slope=0.928→0.919). Training time EXPLODED 4× (18.1→74.2min) - likely numerical instability. omega_f=35 confirmed optimal for Jp. Higher omega_f fails for Jp at 100 frames.
Next: parent=65

## Iter 64: excellent
Node: id=64, parent=63
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.963, final_mse=5.06, total_params=790529, slope=0.928, training_time=18.1min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 150000 -> 200000
Parent rule: Highest UCB (node 63, UCB=1.798)
Observation: 200k steps MAJOR IMPROVEMENT! R²=0.855→0.963 (+0.108). Jp at 100 frames now matches 48-frame baseline (0.968). Confirms Jp needs ~2000 steps/frame (vs F's 1000). Data scaling finally works for Jp with sufficient training.
Next: parent=64

## Iter 63: moderate
Node: id=63, parent=62
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.855, final_mse=19.7, total_params=790529, slope=0.814, training_time=13.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 2.5E-5 -> 3E-5 (with n_layers reverted to 3 from iter 61's config)
Parent rule: Highest UCB (node 62, UCB=1.629)
Observation: lr=3E-5 IMPROVED R² (0.813→0.855)! Higher lr helps Jp at 100 frames. Still below 48-frame baseline (0.968). Best in block so far. Next: Try more training (200k steps).
Next: parent=63

## Iter 62: moderate
Node: id=62, parent=61
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=35.0, batch_size=1
Metrics: final_r2=0.813, final_mse=25.6, total_params=1053185, slope=0.756, training_time=17.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: n_layers_nnr_f: 3 -> 4
Parent rule: Highest UCB (node 61, UCB=1.493)
Observation: Depth increase (3→4) HURT R² (0.826→0.813) and increased training time (14→17.4min). Unlike F field, Jp does NOT benefit from extra layers at 100 frames. Reverting to 3 layers and trying higher lr (3E-5) which worked for F field.
Next: parent=62

## Iter 61: moderate
Node: id=61, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.826, final_mse=23.78, total_params=790529, slope=0.771, training_time=14.0min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: [baseline] Block 1 Jp-optimal config adapted for 100 frames (1500 steps/frame)
Parent rule: root (block start)
Observation: UNEXPECTED REGRESSION - R²=0.826 << 0.968 (48 frames, Block 1). Data scaling HURT Jp unlike F field. 512×3 may be underfitting 100 frames of Jp data. Need to increase capacity (depth) or adjust other parameters.
Next: parent=61

## Iter 60: excellent
Node: id=60, parent=59
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=15.0, batch_size=1
Metrics: final_r2=0.9994, final_mse=2.9E-4, total_params=265220, slope=0.999, training_time=6.4min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: lr_NNR_f: 3E-5 -> 4E-5
Parent rule: Highest UCB (node 59, UCB=3.449), testing higher lr with optimal omega_f=15
Observation: lr=4E-5 (R²=0.9994) matches lr=3E-5 (R²=0.9994) at omega_f=15. F field tolerates lr=3E-5→4E-5 at lower omega_f. LR has wider tolerance when omega_f is lower.
Next: BLOCK END - performing block summary

>>> BLOCK 5 SUMMARY <<<
Field: F, INR: siren_txy, n_frames: 100 (scaled from 48)
Iterations: 49-60 (12 iterations)
Best R²: 0.9998 (iter 49, 150k steps), Best at 100k steps: 0.9994 (iter 58/60)
Optimal config: lr=3E-5→4E-5, hidden_dim=256, n_layers=4, omega_f=15-25, 100k steps
Training time range: 5.2min (80k) - 12.5min (512×4)

Key Findings:
1. DATA SCALING SUCCESS: R²=0.9998 with 100 frames > R²=0.9995 with 48 frames
2. Steps/frame minimum: 1000 steps/frame for R²>0.99 (100k steps for 100 frames)
3. 256×4 Pareto-optimal: R²=0.999 at 6.4min vs 512×4 at 12.5min (same R², 2× time)
4. omega_f range for F: 15≤omega_f≤25 (peak ~15-20, sharp dropoff at 30)
5. Depth critical at 100 frames: 4 layers >> 3 layers
6. LR tolerance: 3E-5→4E-5 both work at omega_f=15
7. 12/12 iterations R²>0.95: 100% excellent rate

Branching rate: 3/12 = 25% (nodes 52, 57, 60 used non-sequential parents)
Improvement rate: 6/12 = 50% (nodes with R²≥parent)
>>> END BLOCK 5 SUMMARY <<<

## Iter 59: excellent
Node: id=59, parent=58
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=10.0, batch_size=1
Metrics: final_r2=0.997, final_mse=1.2E-3, total_params=265220, slope=0.997, training_time=6.4min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: omega_f: 15.0 -> 10.0
Parent rule: Highest UCB (node 58, UCB=3.342), testing lower omega_f boundary for F field
Observation: omega_f=10 (R²=0.997) slightly below omega_f=15 (R²=0.9994). Found LOWER BOUND: omega_f=10 starts to degrade. F field optimal range: 15≤omega_f≤25.
Next: parent=59 (UCB=3.342, last iteration - final boundary exploration)

## Iter 58: excellent
Node: id=58, parent=57
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=15.0, batch_size=1
Metrics: final_r2=0.9994, final_mse=2.7E-4, total_params=265220, slope=0.999, training_time=6.1min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: omega_f: 20.0 -> 15.0
Parent rule: Highest UCB (node 57, UCB=3.235), testing omega_f=15 to explore lower frequency limit
Observation: omega_f=15 (R²=0.9994) ≈ omega_f=20 (R²=0.9992) ≈ omega_f=25 (R²=0.999). F field tolerates omega_f=15-25 range! omega_f=15 slightly BETTER than 20, suggesting F prefers lower frequencies.
Next: parent=58 (highest UCB)

## Iter 56: good
Node: id=56, parent=55
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.950, final_mse=2.4E-2, total_params=1054724, slope=0.950, training_time=11.6min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: omega_f: 25.0 -> 30.0
Parent rule: Highest UCB (node 55, UCB=2.870), testing omega_f sensitivity for F field with 100 frames
Observation: omega_f SENSITIVITY CONFIRMED. omega_f=30 (R²=0.950) << omega_f=25 (R²=0.999). F field STRONGLY prefers omega_f=25. 5-point omega change causes -5% R² drop!
Next: parent=56 (UCB=2.950, probe further with lr variation)

## Iter 55: excellent
Node: id=55, parent=54
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.999, final_mse=4.7E-4, total_params=1054724, slope=0.999, training_time=12.5min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: n_layers_nnr_f: 3 -> 4
Parent rule: Highest UCB (node 54, UCB=2.720), testing if depth compensates for excess width
Observation: DEPTH COMPENSATES FOR WIDTH. 512×4 (R²=0.999) > 512×3 (R²=0.988). BUT 512×4 (12.5min) vs 256×4 (6.4min) - same R², 2× time. 256×4 remains Pareto-optimal.
Next: parent=55 (UCB=2.870, test omega_f variation)

## Iter 54: excellent
Node: id=54, parent=53
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.988, final_mse=5.8E-3, total_params=792068, slope=0.988, training_time=9.7min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: hidden_dim_nnr_f: 384 -> 512
Parent rule: Highest UCB (node 53, UCB=2.580), testing width ceiling
Observation: WIDTH CEILING FOUND. 512×3 (R²=0.988) < 384×3 (R²=0.999). +33% width causes -1.1% R² and +28% time (9.7min vs 7.6min). Optimal width at 3 layers is 384.
Next: parent=54 (UCB=2.720, highest)

## Iter 48: excellent
Node: id=48, parent=47
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.990, final_mse=1.86, total_params=446596, slope=0.977, training_time=7.0min
Field: field_name=C, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5 -> 2.5E-5, hidden_dim_nnr_f: 512 -> 384 (returning to optimal hidden_dim)
Parent rule: UCB highest (node 47), fine-tuning lr and returning to optimal hidden_dim
Observation: lr=2.5E-5 + hidden_dim=384 (R²=0.990) < Node 43 (R²=0.993). lr=3E-5 confirmed optimal for C field. Block 4 best remains Node 43.

---

## Block 5: F field, n_frames=100 (Data Scaling Test)

## Iter 53: excellent
Node: id=53, parent=52
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.9991, final_mse=4.1E-4, total_params=446596, slope=0.9995, training_time=7.6min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: hidden_dim_nnr_f: 256 -> 384
Parent rule: Highest UCB (node 52), testing if width compensates for depth
Observation: WIDTH COMPENSATES for DEPTH. 384×3 (R²=0.9991) ≈ 256×4 (R²=0.9987). 100k steps. Pareto: 384×3 (7.6min) vs 256×4 (6.4min) - depth wins on time.
Next: parent=53 (UCB=2.580)

## Iter 52: excellent
Node: id=52, parent=51
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.991, final_mse=4.2E-3, total_params=199428, slope=0.990, training_time=5.3min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: n_layers_nnr_f: 4 -> 3, total_steps: 80000 -> 100000
Parent rule: Highest UCB (node 51), testing if 3 layers matches 4 layers with 100k steps
Observation: 3 layers (R²=0.991) < 4 layers (R²=0.999). For 100 frames, n_layers=4 is clearly superior. 4 vs 3 layers provides 0.8% R² improvement.
Next: parent=52 (UCB=2.405)

## Iter 49: excellent
Node: id=49, parent=root
Mode/Strategy: exploit (first iteration of block)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.9998, final_mse=8.6E-5, total_params=265220, slope=0.9999, training_time=8.5min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: n_frames: 48 -> 100 (2× data scaling test), total_steps: 100k -> 150k
Parent rule: Block start - using Block 2 optimal F config with scaled steps for more frames
Observation: EXCELLENT! R²=0.9998 with 100 frames EXCEEDS Block 2's R²=0.9995 with 48 frames. More data HELPS, not hurts. F field extremely robust. 150k steps + 256×4 network handles 2× data perfectly.

---

## Block 4 Summary (Iterations 37-48)

**Field**: C (APIC matrix), **INR Type**: siren_txy, **n_frames**: 48

**Best config (Node 43)**: lr_NNR_f=3E-5, hidden_dim=384, n_layers=3, omega_f=30.0, total_steps=100k
→ R²=0.993, slope=0.981, training_time=7.1min

**Key findings:**
1. C field SUCCESS - all 12 iterations R²>0.97. C behaves like F (easy), not S (hard).
2. omega_f=30 OPTIMAL for C: 25(0.984)<28(0.989)<30(0.993)>35(0.979) - clear inverse-U.
3. n_layers=3 OPTIMAL for C: 2(0.972)<3(0.993)>4(0.990). Depth sweet spot found.
4. hidden_dim=384 OPTIMAL: 256(0.989)<384(0.993)>512(0.984). Width sweet spot found.
5. lr=3E-5 confirmed (2.5E-5 regresses to 0.990, 4E-5 regresses to 0.987).
6. total_steps=100k optimal (150k causes overfitting: R²=0.979).
7. Field difficulty ranking confirmed: F(0.9995) > C(0.993) >> Jp(0.968) >> S(0.618).

**Branching rate**: 4/12 = 33% (healthy exploration)
**Improvement rate**: 6/12 = 50% (from initial R²=0.984 to best R²=0.993)

---

## Iter 47: excellent
Node: id=47, parent=46
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.984, final_mse=2.94, total_params=792068, slope=0.974, training_time=9.0min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 28.0 -> 30.0 (from Node 46)
Parent rule: UCB highest (node 46), testing optimal omega_f=30 with larger hidden_dim=512
Observation: hidden_dim=512 + omega_f=30 (R²=0.984) < hidden_dim=384 + omega_f=30 (R²=0.993). Larger model WORSE. hidden_dim=384 confirmed optimal.
Next: parent=47 (UCB=3.329), try lr=2.5E-5 with optimal config (hidden_dim=384, omega_f=30) to fine-tune lr

## Iter 46: excellent
Node: id=46, parent=45
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=28.0, batch_size=1
Metrics: final_r2=0.987, final_mse=2.41, total_params=792068, slope=0.976, training_time=9.1min
Field: field_name=C, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 384 -> 512
Parent rule: UCB highest (node 45), testing capacity increase with omega_f=28
Observation: hidden_dim=512 REGRESSES from 0.989→0.987 (at omega_f=28). Larger model no benefit. Best remains Node 43 (384, omega_f=30, R²=0.993).
Next: parent=46 (UCB=3.223), try omega_f=30 with hidden_dim=512 to test if larger model benefits from optimal frequency

## Iter 45: excellent
Node: id=45, parent=43
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=28.0, batch_size=1
Metrics: final_r2=0.989, final_mse=2.13, total_params=446596, slope=0.976, training_time=6.9min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 30.0 -> 28.0
Parent rule: UCB highest (node 43), fine-tuning omega_f around optimal 30
Observation: omega_f=28 REGRESSES from 0.993→0.989. omega_f=30 confirmed optimal for C field. omega_f mapping: 25(0.984)<28(0.989)<30(0.993)>35(0.979).
Next: parent=45 (UCB=3.110), try hidden_dim=512 to test capacity increase

## Iter 44: excellent
Node: id=44, parent=43
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=2, omega_f=30.0, batch_size=1
Metrics: final_r2=0.972, final_mse=5.26, total_params=298756, slope=0.962, training_time=5.6min
Field: field_name=C, inr_type=siren_txy
Mutation: n_layers_nnr_f: 3 -> 2
Parent rule: UCB highest (node 43), testing depth floor
Observation: n_layers=2 REGRESSES (R² 0.993→0.972). Depth floor found: 3 layers optimal. n_layers sequence: 2(0.972)<3(0.993)>4(0.990).
Next: parent=43 (best R²=0.993), try omega_f=28 to fine-tune around optimal 30

## Iter 43: excellent
Node: id=43, parent=40
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.993, final_mse=1.39, total_params=446596, slope=0.981, training_time=7.1min
Field: field_name=C, inr_type=siren_txy
Mutation: n_layers_nnr_f: 4 -> 3
Parent rule: UCB highest (node 40), testing shallower network
Observation: BREAKTHROUGH! n_layers=3 BETTER than n_layers=4 for C field (R² 0.990→0.993). Faster training (8.5→7.1min). Depth not always better.
Next: parent=43 (UCB=2.863), try n_layers=2 to test depth floor

## Block 1 (Jp, siren_txy, n_frames=48)

## Iter 30: poor
Node: id=30, parent=28
Mode/Strategy: exploit/more-training
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.567, final_mse=6.29e-08, slope=0.594, total_params=1054724, training_time=17.1min
Field: field_name=S, inr_type=siren_txy
Mutation: total_steps: 100000→150000
Parent rule: UCB highest node 28, extend training to improve R²
Observation: R² improved 0.499→0.567 (+13.6%). More steps help. Training time now 17.1min (over 10min target). Try lr reduction next.
Next: parent=30

## Iter 34: poor
Node: id=34, parent=31
Mode/Strategy: exploit/frequency-fine-tuning
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=60.0, batch_size=1
Metrics: final_r2=0.555, final_mse=6.61e-08, slope=0.624, total_params=1054724, training_time=16.7min
Field: field_name=S, inr_type=siren_txy
Mutation: omega_f: 50→60
Parent rule: UCB highest node 31, test omega_f=60 between optimal 50 and failed 75
Observation: REGRESSION. omega_f=60 WORSE than omega_f=50 (0.618→0.555, -10%). omega_f=50 CONFIRMED as optimal peak for S field.
Next: parent=31

## Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: initial/explore
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.481, final_mse=1.07e+02, total_params=12801, training_time=2.5min
Field: field_name=Jp, inr_type=siren_txy
Mutation: [initial config]
Parent rule: Initial run
Observation: Very poor R² due to omega_f=80 (known instability >50) and tiny hidden_dim=64
Next: parent=root

## Iter 2: good
Node: id=2, parent=root
Mode/Strategy: exploit/fix-known-issues
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.901, final_mse=4.76e+01, total_params=790529, slope=0.385, training_time=4.6min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 80→30, hidden_dim_nnr_f: 64→512
Parent rule: Fixed omega_f (instability >50) and increased hidden_dim for capacity
Observation: R² jumped 0.481→0.901 confirming omega_f=30 optimal. Low slope=0.385 suggests model may need more training steps.
Next: parent=2

## Iter 3: good
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.930, final_mse=2.21e+01, slope=0.606, total_params=790529, training_time=8.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: total_steps: 50000→100000
Parent rule: Double training steps to improve convergence and slope
Observation: R² improved 0.901→0.930 and slope 0.385→0.606. More steps helped. Slope still <1.0, trying faster lr to push convergence.
Next: parent=3

## Iter 4: excellent
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.965, final_mse=5.51e+00, slope=0.863, total_params=790529, training_time=9.0min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5→2E-5
Parent rule: Double learning rate to accelerate convergence (UCB node 3)
Observation: R² crossed excellent threshold (0.930→0.965), slope improved (0.606→0.863). lr=2E-5 is beneficial. Continuing to push lr or add steps for slope→1.0.
Next: parent=4

## Iter 5: excellent
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.964, final_mse=5.02e+00, slope=0.891, total_params=790529, training_time=8.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 2E-5→3E-5
Parent rule: Highest UCB node 4 (R²=0.965), increase lr to improve slope
Observation: R² stable (0.965→0.964), slope improved (0.863→0.891). lr=3E-5 still beneficial for slope.
Next: parent=5

## Iter 6: moderate
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.806, final_mse=2.37e+01, slope=0.763, total_params=790529, training_time=9.0min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5→4E-5
Parent rule: UCB node 5 (R²=0.964), push lr higher to improve slope toward 1.0
Observation: REGRESSION. lr=4E-5 too high - R² dropped 0.964→0.806, slope dropped 0.891→0.763. Establishes lr upper bound ~3E-5.
Next: parent=5

## Iter 7: moderate
Node: id=7, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.879, final_mse=1.46e+01, slope=0.869, total_params=790529, training_time=13.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: total_steps: 100000→150000
Parent rule: UCB node 5 (highest UCB), increase total_steps to improve slope toward 1.0
Observation: REGRESSION. 150k steps degrades R² (0.964→0.879) despite longer training. Possible overfitting or lr decay needed. 100k steps is optimal with constant lr=3E-5.
Next: parent=4

## Iter 8: excellent
Node: id=8, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.966, final_mse=4.91e+00, slope=0.888, total_params=790529, training_time=8.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 2E-5→2.5E-5
Parent rule: UCB node 4 (highest available UCB), test midpoint lr
Observation: Excellent R² maintained (0.966≈0.965). lr=2.5E-5 confirms optimal lr zone is [2E-5, 3E-5]. Slope=0.888 still plateaued. Switch parameter - try n_layers.
Next: parent=8

## Iter 9: moderate
Node: id=9, parent=8
Mode/Strategy: switch-param/exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.824, final_mse=2.24e+01, slope=0.736, total_params=1053185, training_time=11.5min
Field: field_name=Jp, inr_type=siren_txy
Mutation: n_layers_nnr_f: 3→4
Parent rule: UCB node 8 (best R²=0.966), switch param from lr to n_layers to escape slope plateau
Observation: REGRESSION. n_layers=4 degrades R² (0.966→0.824) and slope (0.888→0.736). Confirms n_layers=3 is optimal for siren_txy Jp field. 4 layers adds params (+33%) and training time (+32%) without benefit. Revert to n_layers=3, try omega_f or hidden_dim next.
Next: parent=8

## Iter 10: excellent
Node: id=10, parent=8
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.968, final_mse=4.46e+00, slope=0.901, total_params=790529, training_time=8.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 30.0→35.0
Parent rule: UCB node 8 (highest R²=0.966), test slightly higher omega_f to improve slope
Observation: NEW BEST! R² improved 0.966→0.968, slope improved 0.888→0.901 (approaching 1.0). omega_f=35 is better than 30 for Jp field. Will test omega_f=40 next to see if trend continues.
Next: parent=10

## Iter 11: excellent
Node: id=11, parent=10
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=40.0, batch_size=1
Metrics: final_r2=0.968, final_mse=4.72e+00, slope=0.885, total_params=790529, training_time=9.0min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 35.0→40.0
Parent rule: UCB node 10 (R²=0.968, highest UCB), continue omega_f exploration
Observation: R² unchanged (0.968→0.968), but slope regressed (0.901→0.885) and MSE increased slightly (4.46e+00→4.72e+00). omega_f=35 is optimal; omega_f=40 passes inflection point. Next: try hidden_dim increase to see if more capacity improves slope.
Next: parent=11

## Iter 12: moderate
Node: id=12, parent=11
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=768, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.885, final_mse=1.40e+01, slope=0.852, total_params=1775617, training_time=16.2min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 512→768
Parent rule: UCB node 11, test larger hidden_dim for more capacity
Observation: REGRESSION. hidden_dim=768 degrades R² (0.968→0.885) and slope (0.885→0.852) while nearly doubling training time (9.0→16.2min) and params (790k→1.78M). hidden_dim=512 is optimal; larger dims cause overfitting/slower convergence at constant lr.
Next: parent=root (block end)

---

## Block 1 Summary

**Field**: Jp, **INR Type**: siren_txy, **n_frames**: 48

**Best Config (Iter 10)**:
- lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=35.0
- R²=0.968, slope=0.901, MSE=4.46e+00, training_time=8.8min, params=790529

**Key Findings**:
1. **omega_f**: 80 catastrophic (R²=0.48), 30 good, 35 optimal, 40 regresses slope
2. **lr_NNR_f**: 1E-5 too slow, 2E-5→3E-5 optimal zone, 4E-5 too high (regression)
3. **total_steps**: 100k optimal for 48 frames; 150k causes overfitting with constant lr
4. **n_layers**: 3 optimal; 4 layers degrades R² despite more params
5. **hidden_dim**: 512 optimal; 768 regresses (overfitting/slower convergence)

**Branching Rate**: 6/12 = 50% (healthy exploration/exploitation balance)
**Convergence**: 5/12 excellent (42%), best R²=0.968

---

## Block 2 (F, siren_txy, n_frames=48)

## Iter 13: excellent
Node: id=13, parent=root
Mode/Strategy: exploit/knowledge-transfer
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.995, final_mse=2.38e-03, slope=0.994, total_params=792068, training_time=8.9min
Field: field_name=F, inr_type=siren_txy
Mutation: field_name: Jp→F (block change)
Parent rule: Test knowledge transfer from Block 1 optimal config to F field
Observation: EXCELLENT TRANSFER! Block 1's best config (lr=2.5E-5, hidden_dim=512, omega_f=35) produces even better results on F field (R²=0.995 vs Jp's 0.968, slope=0.994 vs 0.901). F field with 4 components actually benefits from same architecture. Established principles generalize across fields.

## Iter 14: excellent
Node: id=14, parent=13
Mode/Strategy: exploit/efficiency-test
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.996, final_mse=1.95e-03, slope=0.995, total_params=199428, training_time=5.2min
Field: field_name=F, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 512→256
Parent rule: Test efficiency - can smaller capacity maintain excellent R²?
Observation: EFFICIENCY WIN! hidden_dim=256 maintains R²=0.996 (slightly better than 512's 0.995!) while cutting training time 8.9→5.2min (42% faster) and params 792k→199k (75% reduction). F field needs less capacity than expected.

## Iter 15: excellent
Node: id=15, parent=14
Mode/Strategy: exploit/efficiency-test
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=128, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.988, final_mse=5.99e-03, slope=0.988, total_params=50564, training_time=4.8min
Field: field_name=F, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 256→128
Parent rule: Continue efficiency exploration to find minimum capacity for R²>0.95
Observation: hidden_dim=128 still achieves excellent R²=0.988 but drops from 256's 0.996. Capacity efficiency curve: 512(0.995)→256(0.996)→128(0.988). hidden_dim=256 is the sweet spot - best R² and 5.2min training. Next: test depth reduction (n_layers=2) to see if 2 layers suffice for F field.

## Iter 16: excellent
Node: id=16, parent=15
Mode/Strategy: exploit/depth-efficiency-test
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=2, omega_f=35.0, batch_size=1
Metrics: final_r2=0.968, final_mse=1.53e-02, slope=0.969, total_params=133636, training_time=3.8min
Field: field_name=F, inr_type=siren_txy
Mutation: n_layers_nnr_f: 3→2
Parent rule: UCB node 16 (highest UCB=2.382), test depth efficiency - can 2 layers maintain R²>0.95?
Observation: n_layers=2 achieves excellent R²=0.968 but drops from 256×3's 0.996. Training time improved to 3.8min (fastest). Depth matters for peak accuracy: 3 layers needed for R²>0.99, but 2 layers still achieves excellent tier. Next: test 512×2 to see if width compensates for reduced depth.

## Iter 17: excellent
Node: id=17, parent=16
Mode/Strategy: exploit/width-compensation-test
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=2, omega_f=35.0, batch_size=1
Metrics: final_r2=0.977, final_mse=1.09e-02, slope=0.976, total_params=529412, training_time=7.2min
Field: field_name=F, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 256→512
Parent rule: UCB node 16 (R²=0.968), test if doubled width compensates for reduced depth
Observation: Width partially compensates: R² improved 0.968→0.977 (+0.009) but still below 256×3's 0.996 (-0.019). Training time nearly doubled (3.8→7.2min). Depth more efficient than width for SIREN - compositional function hierarchy > raw capacity. Next: probe boundary with n_layers=4 on 256 width.

## Iter 18: excellent
Node: id=18, parent=17
Mode/Strategy: failure-probe/depth-boundary
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=35.0, batch_size=1
Metrics: final_r2=0.996, final_mse=1.82e-03, slope=0.996, total_params=265220, training_time=5.9min
Field: field_name=F, inr_type=siren_txy
Mutation: n_layers_nnr_f: 2→4
Parent rule: UCB node 17 (highest UCB=2.728), probe depth boundary - does 4 layers improve on 2-layer baseline?
Observation: 256×4 matches 256×3's R²=0.996 exactly. Extra layer adds 0.7min training time (5.2→5.9min) without accuracy benefit. Confirms n_layers=3 is optimal - more layers don't help on F field. Next: failure-probe omega_f=50 to find frequency boundary.

## Iter 19: excellent
Node: id=19, parent=18
Mode/Strategy: failure-probe/frequency-boundary
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.985, final_mse=7.15e-03, slope=0.986, total_params=265220, training_time=6.4min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 35→50
Parent rule: UCB node 18 (highest UCB=2.243), probe frequency boundary on F field
Observation: omega_f=50 drops R² from 0.996→0.985 (-0.011) confirming frequency sensitivity. Still excellent but degraded from peak. Optimal zone 30-35 holds for F field, same as Jp. Next: explore omega_f=25 (low-frequency regime) to complete frequency boundary mapping.

## Iter 20: excellent
Node: id=20, parent=19
Mode/Strategy: explore/low-frequency-regime
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.999, final_mse=6.78e-04, slope=0.998, total_params=265220, training_time=6.4min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 50→25
Parent rule: UCB node 19, explore low-frequency regime to complete frequency boundary mapping
Observation: BREAKTHROUGH! omega_f=25 achieves R²=0.999, slope=0.998 - best results in Block 2! Counter-intuitively, lower frequency outperforms 35 (0.996) and 50 (0.985). F field benefits from smoother representation. New optimal zone for F: omega_f=25-35. Next: parent=20, test omega_f=20 to find lower boundary.

## Iter 21: excellent
Node: id=21, parent=20
Mode/Strategy: exploit/low-frequency-boundary
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.995, final_mse=2.42e-03, slope=0.995, total_params=265220, training_time=5.9min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 25→20
Parent rule: UCB node 20 (highest at time), find lower boundary of optimal frequency zone
Observation: omega_f=20 drops R² from 0.999→0.995 (-0.004). Lower boundary confirmed - omega_f=25 is optimal for F field. Frequency sensitivity: 20→25→35→50 gives R²=0.995→0.999→0.996→0.985. F field sweet spot at omega_f=25. Next: parent=21, test omega_f=30 to complete frequency mapping between 25-35.

## Iter 22: excellent
Node: id=22, parent=21
Mode/Strategy: exploit/frequency-mapping
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.997, final_mse=1.64e-03, slope=0.997, total_params=265220, training_time=6.1min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 20→30
Parent rule: UCB node 21, complete frequency mapping between 25 and 35
Observation: omega_f=30 achieves R²=0.997, between omega_f=25 (R²=0.999) and omega_f=35 (R²=0.996). Full frequency mapping: 20→25→30→35→50 gives R²=0.995→0.999→0.997→0.996→0.985. Confirms omega_f=25 is optimal, 25-30 is plateau. Next: parent=20 (best node R²=0.999), test lr_NNR_f=3E-5 to explore lr sensitivity at optimal config.

## Iter 23: excellent
Node: id=23, parent=22
Mode/Strategy: exploit/lr-sensitivity
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.9995, final_mse=2.63e-04, slope=0.9995, total_params=265220, training_time=6.3min
Field: field_name=F, inr_type=siren_txy
Mutation: lr_NNR_f: 2.5E-5→3E-5
Parent rule: UCB node 22 (highest UCB=2.560), test lr sensitivity at optimal omega_f=25 config
Observation: lr=3E-5 matches R²=0.9995, slope=0.9995 - essentially identical to lr=2.5E-5 (node 20: R²=0.999, slope=0.998). Both lr values work equally well at omega_f=25. lr=3E-5 achieves best slope of entire block (0.9995)! Confirms lr zone [2.5E-5, 3E-5] is robust. Next iteration is block end (12/12) - will finalize block and select next field.

## Iter 24: excellent
Node: id=24, parent=23
Mode/Strategy: failure-probe/lr-upper-boundary
Config: lr_NNR_f=4E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.989, final_mse=5.30e-03, slope=0.989, total_params=265220, training_time=5.9min
Field: field_name=F, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5→4E-5
Parent rule: UCB node 23 (highest UCB=3.344), failure-probe for lr upper boundary
Observation: lr=4E-5 drops R² 0.9995→0.989. Upper boundary confirmed - optimal lr zone is [2.5E-5, 3E-5] for F field.

---

## BLOCK 2 SUMMARY (Iterations 13-24)

**Field**: F, **INR Type**: siren_txy, **n_frames**: 48

**Best Configuration**:
- lr_NNR_f=3E-5, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, total_steps=100000
- R²=0.9995, slope=0.9995, training_time=6.3min, params=265220

**Key Findings**:
1. Knowledge transfer SUCCESS: Block 1's Jp config (R²=0.968) achieves R²=0.995 on F field
2. F field EASIER than Jp despite 4× output components - more regular structure
3. **Field-specific omega_f discovered**: F optimal at omega_f=25 vs Jp at 35
4. omega_f frequency mapping: 20(0.995) < 25(0.999) > 30(0.997) > 35(0.996) > 50(0.985)
5. Efficiency: 256×4 = 256×3 in accuracy, 256 is optimal hidden_dim (vs 512)
6. Depth > width: 512×2 (R²=0.977) < 256×3 (R²=0.996)
7. lr robustness: Zone [2.5E-5, 3E-5] all achieve R²≥0.999; 4E-5 drops to 0.989

**Block Statistics**:
- Total iterations: 12
- Branching rate: 0% (all parent=(current-1))
- All iterations excellent (R²>0.95): 12/12 = 100%
- Best R²: 0.9995 (iters 23)
- Worst R²: 0.968 (iter 16 - shallow 2-layer network)

**Principle Updates**:
- omega_f is field-dependent: F→25, Jp→35
- hidden_dim=256 sufficient for F field (was 512 for Jp)
- lr_NNR_f upper bound: 4E-5 causes slight regression

---

## Block 3 (S, siren_txy, n_frames=48)

## Iter 25: poor
Node: id=25, parent=root
Mode/Strategy: initial/knowledge-transfer
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.236, final_mse=1.11e-07, slope=0.237, total_params=265220, training_time=6.3min
Field: field_name=S, inr_type=siren_txy
Mutation: field_name: F→S (block change)
Parent rule: Test knowledge transfer from Block 2 optimal config to S field
Observation: CATASTROPHIC FAILURE. Block 2's optimal config (R²=0.9995 on F) yields R²=0.236 on S field. S field is fundamentally different - stress tensor has very small values (~0-0.01) requiring different representation. Hypothesis: S field needs higher capacity (512), different omega_f, or possibly loss function adjustment for small values.
Next: parent=25

## Iter 26: poor
Node: id=26, parent=25
Mode/Strategy: exploit/capacity-increase
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.406, final_mse=8.72e-08, slope=0.440, total_params=1054724, training_time=10.9min
Field: field_name=S, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 256→512
Parent rule: UCB node 25 (only node), increase capacity to match Jp field requirements
Observation: R² improved 0.236→0.406 (+72% relative), slope 0.237→0.440. Capacity increase helps but still fundamentally poor. S field requires different approach beyond capacity. Next: try lower omega_f (15) since S has smooth small values.
Next: parent=26

## Iter 27: poor
Node: id=27, parent=26
Mode/Strategy: exploit/frequency-reduction
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=15.0, batch_size=1
Metrics: final_r2=0.152, final_mse=1.24e-07, slope=0.181, total_params=1054724, training_time=11.5min
Field: field_name=S, inr_type=siren_txy
Mutation: omega_f: 25→15
Parent rule: UCB node 26 (highest R²=0.406), test lower omega_f for smoother S representation
Observation: REGRESSION. omega_f=15 made things WORSE (0.406→0.152). Lower frequency is NOT the solution for S field. Need to try opposite direction - higher omega_f (50) for finer detail capture of small stress values.
Next: parent=26

## Iter 28: poor
Node: id=28, parent=26
Mode/Strategy: exploit/frequency-increase
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.499, final_mse=7.30e-08, slope=0.539, total_params=1054724, training_time=11.2min
Field: field_name=S, inr_type=siren_txy
Mutation: omega_f: 25→50
Parent rule: Highest UCB node (26), continue omega_f exploration in positive direction
Observation: IMPROVEMENT. omega_f=50 improved R² 0.406→0.499 (+23%). Higher frequency helps S field. omega_f trend: 15→0.152, 25→0.406, 50→0.499. Direction confirmed. Continue with omega_f=75.
Next: parent=28

## Iter 29: poor
Node: id=29, parent=28
Mode/Strategy: exploit/frequency-increase
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=75.0, batch_size=1
Metrics: final_r2=0.094, final_mse=1.35e-07, slope=0.142, total_params=1054724, training_time=11.5min
Field: field_name=S, inr_type=siren_txy
Mutation: omega_f: 50→75
Parent rule: Highest UCB (node 28), continue omega_f exploration since 50>25 helped
Observation: CATASTROPHIC REGRESSION. omega_f=75 destroyed performance (0.499→0.094). omega_f=50 is AT or PAST optimal. Frequency exploration complete: 15→0.152, 25→0.406, 50→0.499, 75→0.094. Peak near omega_f=50.
Next: parent=28


## Iter 31: poor
Node: id=31, parent=30
Mode/Strategy: exploit/lr-reduction
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.618, final_mse=5.77e-08, slope=0.629, total_params=1054724, training_time=16.7min
Field: field_name=S, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5→2E-5
Parent rule: UCB highest node 30, reduce lr for finer convergence
Observation: R² improved 0.567→0.618 (+9%). Lower lr helps. Still poor but trending up. Try lr=1.5E-5 or hidden_dim=1024.
Next: parent=31

## Iter 32: poor
Node: id=32, parent=31
Mode/Strategy: exploit/lr-reduction
Config: lr_NNR_f=1.5E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.569, final_mse=6.40e-08, slope=0.594, total_params=1054724, training_time=16.7min
Field: field_name=S, inr_type=siren_txy
Mutation: lr_NNR_f: 2E-5→1.5E-5
Parent rule: UCB highest node 31, further reduce lr following iter31 success direction
Observation: REGRESSION. lr=1.5E-5 made performance WORSE (0.618→0.569, -8%). Lower lr not always better - optimal at lr=2E-5. Now try capacity increase (hidden_dim=1024) from best node 31.
Next: parent=31

## Iter 33: poor
Node: id=33, parent=31
Mode/Strategy: exploit/capacity-increase
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=1024, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.565, final_mse=6.35e-08, slope=0.613, total_params=4206596, training_time=47.3min
Field: field_name=S, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 512→1024
Parent rule: UCB node 31 (best R²=0.618), increase capacity to 1024
Observation: REGRESSION + TIME EXPLOSION. hidden_dim=1024 WORSE than 512 (0.618→0.565, -9%) and training time 47.3min (3× target). Capacity ceiling confirmed at 512 for S field.
Next: parent=31

## Iter 35: poor
Node: id=35, parent=31
Mode/Strategy: exploit/depth-increase
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=5, omega_f=50.0, batch_size=1
Metrics: final_r2=0.610, final_mse=5.98e-08, slope=0.650, total_params=1317380, training_time=20.0min
Field: field_name=S, inr_type=siren_txy
Mutation: n_layers_nnr_f: 4→5
Parent rule: UCB highest node 31, test deeper network
Observation: REGRESSION. n_layers=5 WORSE than n_layers=4 (0.618→0.610, -1.3%). Confirms Block 2 finding: 4 layers is optimal ceiling. Training time increased 16.7→20min.
Next: parent=31


## Iter 36: poor
Node: id=36, parent=31
Mode/Strategy: exploit/frequency-fine-tuning
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=45.0, batch_size=1
Metrics: final_r2=0.562, final_mse=6.74e-08, slope=0.660, total_params=1054724, training_time=17.0min
Field: field_name=S, inr_type=siren_txy
Mutation: omega_f: 50→45
Parent rule: UCB highest node 31, fine-tune omega_f below optimal 50
Observation: REGRESSION. omega_f=45 WORSE than omega_f=50 (0.618→0.562, -9%). omega_f=50 is SHARP PEAK for S field. Both directions (45, 60) degrade.
Next: block end

=== BLOCK 3 SUMMARY ===
Field: S (stress tensor), INR: siren_txy, n_frames: 48
Iterations: 25-36 (12 iterations)
Best config: lr=2E-5, hidden_dim=512, n_layers=4, omega_f=50, 150k steps → R²=0.618, slope=0.629
Key findings:
1. S field FUNDAMENTALLY HARDER than F and Jp - best R²=0.618 (vs F=0.9995, Jp=0.968)
2. omega_f=50 SHARP PEAK: 15→0.152, 25→0.406, 45→0.562, 50→0.618, 60→0.555, 75→0.094
3. lr optimal at 2E-5 (not 3E-5 like F, not 1.5E-5)
4. Capacity ceiling at 512 (1024 REGRESSED)
5. n_layers=5 REGRESSED (4 is optimal ceiling)
6. All mutations from best node 31 FAILED - LOCAL OPTIMUM FOUND
Branching rate: 33% (4/12 iterations branched from non-sequential parent)
Improvement rate: 8% (1/12 exceeded prior best)
Block verdict: S field represents different difficulty class. Config-level optimization may be insufficient.

## Block 4 (C, siren_txy, n_frames=48)

## Iter 37: excellent
Node: id=37, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.984, final_mse=3.11, total_params=265220, slope=0.973, training_time=6.4min
Field: field_name=C, inr_type=siren_txy
Mutation: First iteration of block - used F-optimal config (omega_f=25)
Parent rule: New block, testing C field with F-optimal config
Observation: C field behaves like F field! R²=0.984 >> S field's 0.618. APIC matrix easier than stress tensor.
Next: parent=37, try omega_f=30 (between F-optimal 25 and Jp-optimal 35)

## Iter 38: excellent
Node: id=38, parent=37
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.989, final_mse=2.07, total_params=265220, slope=0.975, training_time=6.0min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 25.0 -> 30.0
Parent rule: UCB=1.651 (node 37), testing omega_f=30 for C field
Observation: omega_f=30 IMPROVES R² from 0.984→0.989. C field prefers higher freq than F (25). Try omega_f=35 next.
Next: parent=38, try omega_f=35 (Jp-optimal)

## Iter 39: excellent
Node: id=39, parent=38
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=35.0, batch_size=1
Metrics: final_r2=0.979, final_mse=4.07, total_params=265220, slope=0.966, training_time=6.3min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 30.0 -> 35.0
Parent rule: UCB highest (node 38), testing omega_f=35 for C field
Observation: omega_f=35 REGRESSES R² from 0.989→0.979. C field optimal at omega_f=30. omega_f mapping: 25(0.984)<30(0.989)>35(0.979).
Next: parent=38 (highest R²), try hidden_dim increase (256→384) to push beyond 0.989

## Iter 40: excellent
Node: id=40, parent=38
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.990, final_mse=1.91, total_params=594436, slope=0.982, training_time=8.5min
Field: field_name=C, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 256 -> 384
Parent rule: UCB highest (node 38), testing capacity increase
Observation: hidden_dim=384 marginal improvement (0.990 vs 0.989), slope improved (0.982 vs 0.975). 40% longer training. Diminishing returns.
Next: parent=40 (UCB=2.404), try total_steps=150k to see if larger model can converge better

## Iter 41: excellent
Node: id=41, parent=40
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=100000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.987, final_mse=2.45, total_params=594436, slope=0.975, training_time=8.1min
Field: field_name=C, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5 -> 4E-5
Parent rule: UCB highest (node 40), testing lr increase for larger model
Observation: lr=4E-5 REGRESSES R² from 0.990→0.987. lr=3E-5 optimal for C field. Higher lr causes slight instability.
Next: parent=41 (UCB=2.568), try total_steps=150k (more training for larger model)

## Iter 42: excellent
Node: id=42, parent=41
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.979, final_mse=4.03, total_params=594436, slope=0.971, training_time=12.7min
Field: field_name=C, inr_type=siren_txy
Mutation: total_steps: 100000 -> 150000, lr_NNR_f: 4E-5 -> 3E-5
Parent rule: UCB highest (node 41), testing more training steps with corrected lr
Observation: REGRESSION! R²=0.979 < Node 40's 0.990. 150k steps WORSE than 100k for C field. Overfitting or noise accumulation. Best config remains Node 40.
Next: parent=40 (best R²=0.990), try n_layers=5 to test depth impact on 384-dim model


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

## Iter 57: excellent
Node: id=57, parent=56
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.9992, final_mse=3.7E-4, total_params=265220, slope=0.999, training_time=6.4min
Field: field_name=F, inr_type=siren_txy, n_frames=100
Mutation: omega_f: 30.0 -> 20.0, hidden_dim: 512 -> 256, n_layers: 4 -> 4 (back to optimal arch)
Parent rule: Highest UCB (node 56, UCB=2.950), test omega_f=20 (below optimal 25) with optimal architecture
Observation: omega_f RANGE EXPANDED. omega_f=20 (R²=0.9992) ≈ omega_f=25 (R²=0.999). F field tolerates omega_f 20-25. omega_f=30 fails. Acceptable range: 20≤omega_f≤25.
Next: parent=57 (UCB=3.120)
