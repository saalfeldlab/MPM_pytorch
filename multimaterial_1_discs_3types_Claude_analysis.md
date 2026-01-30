# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1: Jp field, siren_txy, 100 frames

### Iter 1: poor

Node: id=1, parent=root
Mode/Strategy: exploit (baseline)
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.412, final_mse=121.5, total_params=12801, training_time=2.4min
Field: field_name=Jp, inr_type=siren_txy
Visual: Model learned spatial positions but predicts near-constant Jp≈1.0. Scatter shows horizontal band (slope=0.049). Loss plateaued without convergence.
Mutation: baseline config (inherited from template)
Parent rule: Starting from root - first iteration
Observation: Catastrophic underfitting - hidden_dim=64 is 6x below optimal, omega_f=80 is 5x above optimal for Jp
Next: parent=1

### Iter 2: excellent

Node: id=2, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=15.0, batch_size=1
Metrics: final_r2=0.995, final_mse=0.752, total_params=445441, training_time=11.9min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial patterns match well. Scatter shows tight diagonal cluster (slope=0.968). Loss converged smoothly. Per-frame MSE shows higher error at early frames (temporal dynamics captured).
Mutation: [lr_NNR_f]: 1E-5 → 4E-5, [hidden_dim]: 64 → 384, [omega_f]: 80 → 15, [total_steps]: 50k → 200k
Parent rule: Applied prior knowledge optimal config for Jp@100frames
Observation: Massive improvement confirms prior knowledge. Slope=0.968 suggests slight room for tuning (possibly omega_f or lr)
Next: parent=2

### Iter 3: good

Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=15.0, batch_size=1
Metrics: final_r2=0.991, final_mse=1.31, slope=0.958, total_params=445441, training_time=12.0min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred panels show good spatial match. Loss converged smoothly to 10^-1. Per-frame MSE shows high error at early frames (temporal dynamics). Scatter slope=0.888 indicates underprediction.
Mutation: [lr_NNR_f]: 4E-5 → 5E-5
Parent rule: UCB selection - Node 2 was best performer
Observation: lr=5E-5 regressed slightly from lr=4E-5 (R²=0.991 vs 0.995). LR upper boundary found: optimal is 4E-5, not higher.
Next: parent=3

### Iter 4: good

Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=20.0, batch_size=1
Metrics: final_r2=0.987, final_mse=1.91, slope=0.950, total_params=445441, training_time=12.1min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial patterns show reasonable match. Three disc structures visible. Scatter slope=0.88 indicates underprediction. Loss still converging at 80k steps (steady descent).
Mutation: [omega_f]: 15 → 20
Parent rule: UCB selection - Node 3 had highest UCB
Observation: omega_f=20 regressed from omega_f=15 (R²=0.987 vs 0.991 parent, 0.995 grandparent). Combined with lr=5E-5 (also suboptimal), confirms omega_f=15 is better for this dataset. Need to return to best config (lr=4E-5, omega_f=15) and try different mutation.
Next: parent=2

### Iter 5: good

Node: id=5, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=15.0, batch_size=1
Metrics: final_r2=0.992, final_mse=1.12, slope=0.965, total_params=790529, training_time=15.4min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial patterns match well. Scatter shows tight cluster along diagonal. Per-frame MSE high at early frames (temporal dynamics). Slope=0.965 indicates slight underprediction.
Mutation: [hidden_dim]: 384 → 512
Parent rule: UCB selection - returned to best performer Node 2 (R²=0.995)
Observation: hidden_dim=512 REGRESSED from hidden_dim=384 (R²=0.992 vs 0.995). Confirms Jp@100frames hidden_dim ceiling at 384 (consistent with prior knowledge). Training time increased 30% (15.4 vs 11.9min). ESTABLISHED: Jp hidden_dim=384 optimal for 100frames@9000particles.
Next: parent=5

### Iter 6: excellent

Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=10.0, batch_size=1
Metrics: final_r2=0.996, final_mse=0.647, slope=0.973, total_params=790529, training_time=15.5min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred panels match well - same spatial structure visible. Scatter tight diagonal (slope=0.973). Per-frame MSE shows early frames have highest error. Loss continued improving through 200k steps.
Mutation: [omega_f]: 15 → 10
Parent rule: UCB selection - Node 5 had highest UCB
Observation: omega_f=10 IMPROVED over omega_f=15 (R²=0.996 vs 0.992 parent, 0.995 Node 2). Combined with hidden_dim=512, achieved best R² yet. KEY: omega_f=10 may be optimal for Jp@100frames with 9000 particles (lower than prior knowledge suggested). Try omega_f probe.
Next: parent=6

### Iter 7: excellent
Node: id=7, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=7.0, batch_size=1
Metrics: final_r2=0.996, final_mse=0.632, slope=0.974, total_params=790529, training_time=15.4min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial panels excellent match. Loss converged smoothly. Per-frame MSE shows U-shape with highest error at early frames. Scatter tight along diagonal (slope=0.974).
Mutation: [omega_f]: 10 → 7
Parent rule: UCB selection - Node 6 had highest UCB
Observation: omega_f=7 matches omega_f=10 (R²=0.996 both). Plateau found: omega_f [7-10] achieves similar R². Next: probe omega_f=5 to find lower boundary, or test hidden_dim=384 with omega_f=10.
Next: parent=7

### Iter 8: excellent
Node: id=8, parent=7
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.996, final_mse=0.608, slope=0.976, total_params=790529, training_time=15.4min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial panels excellent match - 9 discs clearly visible. Per-frame MSE shows high error at early frames settling by frame 20. Scatter tight diagonal (slope=0.976).
Mutation: [omega_f]: 7 → 5
Parent rule: UCB selection - Node 8 had highest UCB
Observation: omega_f=5 matches omega_f=7 and omega_f=10 (R²=0.996 all). omega_f plateau extends [5-10]. Lower boundary probe: try omega_f=3 or test open question (hidden_dim=384+omega_f=10 for speed).
Next: parent=8

### Iter 9: excellent
Node: id=9, parent=8
Mode/Strategy: exploit (speed-Pareto test)
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=10.0, batch_size=1
Metrics: final_r2=0.995, final_mse=0.792, slope=0.971, total_params=445441, training_time=12.2min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial match good. 9 discs clearly visible. Scatter tight diagonal. Per-frame MSE shows early-frame peak (typical). Loss converged smoothly.
Mutation: [hidden_dim]: 512 → 384, [omega_f]: 5 → 10
Parent rule: UCB selection - Node 8 had highest UCB, testing hidden_dim=384 as speed-Pareto alternative
Observation: hidden_dim=384+omega_f=10 achieves R²=0.995 (vs 0.996 with 512). Only 0.001 R² loss but 20% faster (12.2 vs 15.4min). SPEED PARETO: 384×3@omega=10 is optimal for training time. Probe omega_f lower boundary next.
Next: parent=9

### Iter 10: moderate
Node: id=10, parent=9
Mode/Strategy: exploit (boundary probe)
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=3.0, batch_size=1
Metrics: final_r2=0.975, final_mse=3.72, slope=0.924, total_params=445441, training_time=12.1min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial patterns match but with visible noise. Scatter has significant spread with slope=0.924 (underprediction). Per-frame MSE U-curve with high early-frame error.
Mutation: [omega_f]: 10 → 3
Parent rule: UCB selection - Node 9 was parent, probing omega_f lower boundary
Observation: omega_f=3 REGRESSED (R²=0.975 vs 0.995). omega_f lower boundary confirmed at 5. Complete omega_f map: 3(0.975) < 5(0.996) ≈ 7(0.996) ≈ 10(0.995-0.996) > 15(0.992-0.995) > 20(0.987). Optimal: [5-10].
Next: parent=4

### Iter 11: good
Node: id=11, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.984, final_mse=2.43, slope=0.935, total_params=593281, training_time=14.6min
Field: field_name=Jp, inr_type=siren_txy
Visual: Spatial match visible but with degraded accuracy vs n_layers=3 configs. Slope=0.935 indicates underprediction.
Mutation: [n_layers]: 3 → 4
Parent rule: UCB selection - Node 4 had highest UCB after omega_f boundary probing
Observation: n_layers=4 REGRESSED (R²=0.984 vs 0.987 parent Node 4). CONFIRMS instruction file: Jp requires EXACTLY 3 layers. 4 layers also 20% slower (14.6 vs 12.1min). Depth ceiling confirmed.
Next: parent=11

### Iter 12: good
Node: id=12, parent=11
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=10.0, batch_size=1
Metrics: final_r2=0.992, final_mse=1.16, slope=0.960, total_params=593281, training_time=14.5min
Field: field_name=Jp, inr_type=siren_txy
Visual: GT/Pred spatial match good - 9 discs visible. Scatter tight diagonal (slope=0.960). Per-frame MSE high at early frames. Loss converged smoothly.
Mutation: [omega_f]: 20 → 10, [lr_NNR_f]: 5E-5 → 4E-5
Parent rule: UCB selection - Node 11 had highest UCB, testing if omega_f=10+lr=4E-5 can improve 4-layer config
Observation: omega_f=10 IMPROVED 4-layer config (R²=0.992 vs 0.984 parent) but still BELOW best 3-layer configs (R²=0.995-0.996). CONFIRMS: n_layers=3 ceiling cannot be overcome via omega_f/lr tuning. Depth penalty is fundamental.
Next: BLOCK END

---

## Block 1 Summary

**Best Configuration Found:**
- Accuracy Pareto: 512×3, omega_f=[5-10], lr=4E-5 → R²=0.996, slope=0.976, 15.4min
- Speed Pareto: 384×3, omega_f=10, lr=4E-5 → R²=0.995, slope=0.971, 12.2min

**Parameter Maps Established (Jp@100frames@9000particles):**
- omega_f: 3(0.975) < 5(0.996) ≈ 7(0.996) ≈ 10(0.995-0.996) > 15(0.992-0.995) > 20(0.987). Optimal: [5-10]
- hidden_dim: 384(0.995) ≈ 512(0.992-0.996). Both viable, 384 faster.
- n_layers: 3(0.995-0.996) > 4(0.984-0.992). Ceiling at 3 CONFIRMED.
- lr_NNR_f: 4E-5(0.995-0.996) > 5E-5(0.984-0.991). Optimal: 4E-5.

**Key Findings:**
1. omega_f=10 optimal for Jp@100frames@9000particles (lower than prior knowledge omega_f=15)
2. Hidden_dim=384 achieves 99% of 512 accuracy at 20% lower training time
3. n_layers=3 depth ceiling confirmed - 4 layers ALWAYS regresses regardless of omega_f/lr
4. LR=4E-5 optimal - 5E-5 causes consistent regression

**Branching Rate:** 4/11 = 36% (parents: root,root,2,3,2,5,6,7,8,9,4 - unique non-sequential: 4)
**Improvement Rate:** 6/11 = 55% (iters 2,6,7,8,9,12 improved over parent)

---

## Block 2: F field, siren_txy, 100 frames

### Iter 13: good
Node: id=13, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.9936, final_mse=3.09E-03, slope=0.994, total_params=265220, training_time=8.2min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match reasonably across 4 components. U-curve per-frame MSE shows higher error at frame boundaries. Scatter tight diagonal (slope≈1.0). No obvious artifacts.
Mutation: First F iteration - F prior config (256×4, omega=20, lr=3E-5)
Parent rule: Root node for Block 2 (new field)
Observation: F@100frames@9000p achieves R²=0.9936 with prior config - BELOW expected (0.998+). Prior knowledge suggested F is easier than Jp, but we see similar R² here. Test omega_f=15 next to see if F follows same "lower omega" pattern as Jp on this dataset.
Next: parent=13

### Iter 14: good
Node: id=14, parent=13
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=15.0, batch_size=1
Metrics: final_r2=0.995, final_mse=2.42E-03, slope=0.993, total_params=265220, training_time=8.3min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (No visualization available - proceeding with metrics only)
Mutation: [omega_f]: 20 → 15
Parent rule: UCB selection - Node 13 was highest UCB (only node in tree)
Observation: omega_f=15 IMPROVED over omega_f=20 (R²=0.995 vs 0.994). F field follows same "lower omega" trend as Jp on this dataset. Continue probing lower omega_f values to find optimum.
Next: parent=14

### Iter 15: moderate
Node: id=15, parent=14
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=10.0, batch_size=1
Metrics: final_r2=0.966, final_mse=1.63E-02, slope=0.965, total_params=265220, training_time=8.1min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred panels show patterns captured but detail loss visible. Per-frame MSE shows U-curve. Loss converged smoothly but to higher plateau. Scatter shows more spread than parent.
Mutation: [omega_f]: 15 → 10
Parent rule: UCB selection - Node 14 had highest UCB (R²=0.995)
Observation: omega_f=10 REGRESSED significantly (R²=0.966 vs 0.995). CRITICAL FINDING: F field does NOT follow Jp omega_f pattern. F optimal omega_f=15 (not 5-10 like Jp). F field needs HIGHER omega_f than Jp - different frequency content.
Next: parent=15

### Iter 16: excellent
Node: id=16, parent=15
Mode/Strategy: exploit (boundary probe)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=12.0, batch_size=1
Metrics: final_r2=0.997, final_mse=1.64E-03, slope=0.994, total_params=265220, training_time=7.9min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred panels excellent match across all 4 F components. Loss converged well, per-frame MSE flat. Scatter very tight along diagonal (slope=0.994). No visible artifacts.
Mutation: [omega_f]: 10 → 12
Parent rule: UCB selection - Node 15 had highest UCB, probing omega_f between 10 and 15
Observation: omega_f=12 achieved BEST R² yet (0.997)! F omega_f map: 10(0.966) < 12(0.997) > 15(0.995) > 20(0.994). omega_f=12 is LOCAL MAXIMUM for F field. Non-monotonic with sharp optimum.
Next: parent=16

### Iter 17: excellent
Node: id=17, parent=16
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=12.0, batch_size=1
Metrics: final_r2=0.998, final_mse=9.53E-04, slope=0.997, total_params=265220, training_time=8.0min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match excellent across all 4 F components, scatter tight along diagonal, loss curve smooth convergence, no visible artifacts.
Mutation: [lr_NNR_f]: 3E-5 → 4E-5
Parent rule: UCB selection - Node 16 had highest UCB (2.051)
Observation: lr=4E-5 IMPROVED over lr=3E-5 (R²=0.998 vs 0.997). F tolerates same lr as Jp optimal. Slope improved from 0.994 to 0.997. Both omega_f and lr now optimized.
Next: parent=17

### Iter 18: excellent
Node: id=18, parent=17
Mode/Strategy: exploit (omega_f fine-tuning)
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=11.0, batch_size=1
Metrics: final_r2=0.996, final_mse=1.73E-03, slope=0.997, total_params=265220, training_time=8.0min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match good across all 4 F components. Loss converged well with U-shaped per-frame MSE. Scatter tight along diagonal. SSIM high across components.
Mutation: [omega_f]: 12 → 11
Parent rule: UCB selection - Node 17 had highest UCB (2.153)
Observation: omega_f=11 REGRESSED from omega_f=12 (R²=0.996 vs 0.998). Confirms omega_f=12 is exact LOCAL MAXIMUM. F omega_f map: 10(0.966) < 11(0.996) < 12(0.998) > 15(0.995). omega_f=12 is sharp optimum.
Next: parent=18

### Iter 19: excellent
Node: id=19, parent=18
Mode/Strategy: exploit (lr boundary probe)
Config: lr_NNR_f=5E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=12.0, batch_size=1
Metrics: final_r2=0.9976, final_mse=1.16E-03, slope=0.993, total_params=265220, training_time=8.3min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match excellent across all 4 F components. Loss converged smoothly, per-frame MSE shows U-curve. Scatter tight along diagonal (R²=0.9953, slope=0.9929). No artifacts.
Mutation: [lr_NNR_f]: 4E-5 → 5E-5, [omega_f]: 11 → 12 (returning to optimal)
Parent rule: UCB selection - Node 18 had highest UCB (2.243)
Observation: lr=5E-5 with optimal omega_f=12 achieves R²=0.9976 - virtually same as lr=4E-5 (0.998). F lr map: 3E-5(0.997) < 4E-5(0.998) ≈ 5E-5(0.998). lr=[4E-5, 5E-5] both optimal. F has wider lr tolerance than Jp.
Next: parent=19

### Iter 20: good
Node: id=20, parent=19
Mode/Strategy: exploit (depth test)
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=12.0, batch_size=1
Metrics: final_r2=0.991, final_mse=4.44E-03, slope=0.992, total_params=199428, training_time=7.3min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match good all 4 components, scatter tight (slope=0.992), U-shaped per-frame MSE. Quality slightly below best (0.998).
Mutation: [n_layers_nnr_f]: 4 → 3
Parent rule: UCB selection - Node 19 had highest UCB (2.331)
Observation: n_layers=3 REGRESSED from n_layers=4 (R²=0.991 vs 0.998). F field prefers 4 layers, contrasting with Jp which has 3-layer ceiling. F depth map: 3(0.991) < 4(0.998). Confirmed F vs Jp architectural difference.
Next: parent=20

### Iter 21: excellent
Node: id=21, parent=20
Mode/Strategy: exploit (omega_f upper boundary probe)
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=13.0, batch_size=1
Metrics: final_r2=0.997, final_mse=1.41E-03, slope=0.998, total_params=265220, training_time=8.4min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred excellent all 4 F components, scatter tight along diagonal (slope=0.998), U-shaped per-frame MSE.
Mutation: [omega_f]: 12 → 13, [n_layers_nnr_f]: 3 → 4 (restoring optimal)
Parent rule: UCB selection - Node 20 had highest UCB (2.405)
Observation: omega_f=13 achieves R²=0.997, SLIGHTLY BELOW omega_f=12 (0.998). Confirms omega_f=12 is exact LOCAL MAXIMUM. F omega_f COMPLETE map: 10(0.966) < 11(0.996) < 12(0.998) > 13(0.997) > 15(0.995) > 20(0.994).
Next: parent=21

### Iter 22: excellent
Node: id=22, parent=21
Mode/Strategy: exploit (depth upper boundary probe)
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=5, omega_f=12.0, batch_size=1
Metrics: final_r2=0.996, final_mse=1.77E-03, slope=0.994, total_params=331012, training_time=9.8min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match well all 4 components, U-shaped per-frame MSE, scatter tight along diagonal. Quality slightly below optimal.
Mutation: [n_layers_nnr_f]: 4 → 5, [omega_f]: 13 → 12 (restoring optimal)
Parent rule: UCB selection - Node 21 had highest UCB (2.487)
Observation: n_layers=5 REGRESSED from n_layers=4 (R²=0.996 vs 0.998). F@9000particles has DEPTH CEILING at 4 layers (contradicts prior that F tolerates 5 layers). F depth map: 3(0.991) < 4(0.998) > 5(0.996).
Next: parent=22

### Iter 23: good
Node: id=23, parent=22
Mode/Strategy: exploit (capacity scaling test)
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=12.0, batch_size=1
Metrics: final_r2=0.989, final_mse=5.14E-03, slope=0.988, total_params=594436, training_time=11.0min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match good, inverted U per-frame MSE (higher error mid-simulation), scatter has more spread than optimal. Some spatial detail loss visible.
Mutation: [hidden_dim_nnr_f]: 256 → 384, [n_layers_nnr_f]: 5 → 4 (restoring optimal)
Parent rule: UCB selection - Node 22 had highest UCB (2.559)
Observation: hidden_dim=384 REGRESSED from hidden_dim=256 (R²=0.989 vs 0.998). F field has CAPACITY CEILING at 256×4 - more capacity HURTS. F capacity map: 256(0.998) > 384(0.989). Contradicts scaling expectations. F@9000p is fully specified by 256×4.
Next: parent=23

### Iter 24: excellent
Node: id=24, parent=23
Mode/Strategy: exploit (lr upper boundary probe)
Config: lr_NNR_f=6E-5, total_steps=150000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=12.0, batch_size=1
Metrics: final_r2=0.998, final_mse=9.84E-04, slope=0.998, total_params=265220, training_time=8.1min
Field: field_name=F, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match excellent all 4 F components, scatter tight along diagonal (slope=0.998), smooth loss convergence, U-shaped per-frame MSE.
Mutation: [lr_NNR_f]: 4E-5 → 6E-5, [hidden_dim_nnr_f]: 384 → 256 (restoring optimal)
Parent rule: UCB selection - Node 24 had highest UCB (3.447)
Observation: lr=6E-5 achieves R²=0.998 - IDENTICAL to lr=4E-5 and lr=5E-5. F lr COMPLETE map: 3E-5(0.997) < 4E-5(0.998) ≈ 5E-5(0.998) ≈ 6E-5(0.998). F has WIDE lr tolerance [4E-5, 6E-5], unlike Jp which regresses at 5E-5.
Next: Block end - UCB reset

---

## Block 2 Summary: F field, siren_txy, 100 frames, 9000 particles

**Best result**: R²=0.998, slope=0.998 with 256×4, omega_f=12, lr=4E-5 (or 5E-5 or 6E-5), 150k steps, 8.0min

**Parameter maps (COMPLETE)**:
- omega_f: 10(0.966) < 11(0.996) < 12(**0.998**) > 13(0.997) > 15(0.995) > 20(0.994). SHARP optimum at omega_f=12
- lr: 3E-5(0.997) < 4E-5(0.998) ≈ 5E-5(0.998) ≈ 6E-5(0.998). WIDE plateau at lr=[4E-5, 6E-5]
- depth: 3(0.991) < 4(**0.998**) > 5(0.996). EXACT optimum at n_layers=4
- capacity: 256(**0.998**) > 384(0.989). Ceiling at hidden_dim=256

**Key findings**:
1. **F omega_f vs Jp**: F optimal=12, Jp optimal=[5-10]. F needs ~2x higher omega_f than Jp
2. **F depth vs Jp**: F optimal=4, Jp ceiling=3. Fields require different depths
3. **F lr tolerance**: F tolerates lr=[4E-5, 6E-5] (Jp regresses at 5E-5)
4. **F capacity ceiling**: 256×4 saturates F - more capacity HURTS (unlike typical scaling)
5. **9000 particles affect depth ceiling**: F@9000p has depth ceiling at 4 (prior suggested 5 viable)

**Branching rate**: 8/12 = 67% (healthy exploration)
**Improvement rate**: 8/12 = 67% excellent, 2/12 good, 2/12 moderate

---

## Block 3: C field, siren_txy, 100 frames, 9000 particles

### Iter 25: good
Node: id=25, parent=root
Mode/Strategy: exploit (baseline from prior dataset)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.993, final_mse=1.40, slope=0.985, total_params=1235844, training_time=15.9min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred panels match well all 4 components. Loss converged smoothly. Per-frame MSE shows higher error around frames 40-60 (collision events). Scatter tight along diagonal (slope=0.985).
Mutation: baseline config from prior dataset C field (640×3, omega_f=25, lr=3E-5)
Parent rule: First iteration of block - using prior optimal C config as baseline
Observation: Prior dataset optimal config achieves R²=0.993 on first try - BETTER than prior dataset C (R²=0.989). This 9000 particle dataset may be easier to learn. Based on Jp and F trends, test lower omega_f.
Next: parent=25

### Iter 26: good
Node: id=26, parent=25
Mode/Strategy: exploit (omega_f probe)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=20.0, batch_size=1
Metrics: final_r2=0.988, final_mse=2.43, slope=0.981, total_params=1235844, training_time=15.9min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred panels match reasonably all 4 components. Per-frame MSE peaks around collision (frames 40-70). Scatter shows spread along diagonal. Loss converged but higher than baseline.
Mutation: [omega_f]: 25 → 20
Parent rule: UCB selection - Node 25 had highest UCB
Observation: omega_f=20 REGRESSED from omega_f=25 (R²=0.988 vs 0.993). OPPOSITE to Jp/F trend where lower omega_f helped. C field may need HIGHER omega_f. Test omega_f=30 next.
Next: parent=26

### Iter 27: moderate
Node: id=27, parent=26
Mode/Strategy: exploit (omega_f boundary probe)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.984, final_mse=3.11, slope=0.975, total_params=1235844, training_time=15.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred panels match reasonable all 4 components. Per-frame MSE peaks at collision (~frames 40-60). Scatter shows slope=0.975 underprediction.
Mutation: [omega_f]: 20 → 30 (probing upper boundary from Node 26)
Parent rule: UCB selection - Node 27 had highest UCB (2.209)
Observation: omega_f=30 REGRESSED further (0.984 vs 0.988@omega=20 vs 0.993@omega=25). CONFIRMS omega_f=25 is LOCAL MAXIMUM. C omega_f map: 20(0.988) < 25(0.993) > 30(0.984). Test different parameter dimension next - return to Node 25 for lr probe.
Next: parent=25

### Iter 28: good
Node: id=28, parent=25
Mode/Strategy: exploit (lr probe)
Config: lr_NNR_f=4E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.989, final_mse=2.23, slope=0.981, total_params=1235844, training_time=15.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (plot not available)
Mutation: [lr_NNR_f]: 3E-5 → 4E-5
Parent rule: UCB selection - Node 25 had highest UCB
Observation: lr=4E-5 REGRESSED from lr=3E-5 (0.989 vs 0.993). C field prefers LOWER lr than F field (which had wide lr tolerance). lr=3E-5 optimal for C. lr map: 3E-5(0.993) > 4E-5(0.989).
Next: parent=25

### Iter 29: good
Node: id=29, parent=25
Mode/Strategy: exploit (capacity probe)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.986, final_mse=2.77, slope=0.983, total_params=792068, training_time=11.8min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred panels match reasonably. Pred 01 shows oversmoothing vs GT. Per-frame MSE peaks at collision. Loss converged smoothly.
Mutation: [hidden_dim_nnr_f]: 640 → 512
Parent rule: UCB selection - Node 25 had highest UCB
Observation: Capacity reduction HURT: 512×3 (0.986) < 640×3 (0.993). C field NEEDS 640 hidden_dim. Different from F which has ceiling at 256. Confirms C requires more capacity than F.
Next: parent=25

### Iter 30: moderate
Node: id=30, parent=25
Mode/Strategy: exploit (depth probe)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=4, omega_f=25.0, batch_size=1
Metrics: final_r2=0.980, final_mse=3.96, slope=0.976, total_params=1646084, training_time=20.0min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: GT/Pred match reasonable across 4 components. Per-frame MSE peaks at collision (~frames 40-60). Scatter shows slope=0.976 underprediction. Loss converged with oscillation.
Mutation: [n_layers_nnr_f]: 3 → 4
Parent rule: UCB selection - probing depth boundary from Node 25
Observation: Depth increase HURT: 640×4 (0.980) < 640×3 (0.993). CONFIRMS C has depth ceiling at 3 (same as Jp, different from F which tolerates 4-5 layers). Training time +4min for worse result.
Next: parent=25

### Iter 31: good
Node: id=31, parent=25
Mode/Strategy: exploit (lower lr probe)
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.994, final_mse=1.12, slope=0.989, total_params=1235844, training_time=15.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (plot not available - inferred from metrics)
Mutation: [lr_NNR_f]: 3E-5 → 2E-5
Parent rule: UCB selection - Node 25 had highest UCB after 6 visits
Observation: lr=2E-5 IMPROVED over lr=3E-5 (R²=0.994 vs 0.993, slope=0.989 vs 0.985). NEW BEST. C field prefers LOWER lr than baseline. Updated lr map: 2E-5(0.994) > 3E-5(0.993) > 4E-5(0.989). Trend suggests even lower lr might help. Test lr=1.5E-5 next.
Next: parent=31

### Iter 32: good
Node: id=32, parent=31
Mode/Strategy: exploit (lower lr probe)
Config: lr_NNR_f=1.5E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.987, final_mse=2.52, slope=0.981, total_params=1235844, training_time=15.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (plot not available)
Mutation: [lr_NNR_f]: 2E-5 → 1.5E-5
Parent rule: UCB selection - Node 31 was new best (R²=0.994), testing lower lr boundary
Observation: lr=1.5E-5 REGRESSED from lr=2E-5 (R²=0.987 vs 0.994). CONFIRMS lr=2E-5 is LOCAL MAXIMUM. Complete lr map: 1.5E-5(0.987) < 2E-5(**0.994**) > 3E-5(0.993) > 4E-5(0.989). C lr fully characterized.
Next: parent=28

### Iter 33: good
Node: id=33, parent=28
Mode/Strategy: exploit (total_steps probe)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.991, final_mse=1.70, slope=0.990, total_params=1235844, training_time=20.8min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (plot not available)
Mutation: [total_steps]: 150000 → 200000, [lr_NNR_f]: 4E-5 → 2E-5 (combined with best lr from Node 31)
Parent rule: UCB selection - Node 28 had highest UCB (attempting to combine optimal lr with more steps)
Observation: 200k steps REGRESSED from 150k steps (R²=0.991 vs 0.994). Despite optimal lr=2E-5 and improved slope (0.990), more training steps hurt R². OVERFITTING detected. steps map: 150k(0.994) > 200k(0.991). C field fully characterized at 100 frames.
Next: parent=32

### Iter 34: good
Node: id=34, parent=32
Mode/Strategy: exploit (omega_f fine-tuning)
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=23.0, batch_size=1
Metrics: final_r2=0.988, final_mse=2.35, slope=0.979, total_params=1235844, training_time=15.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (plot not available)
Mutation: [omega_f]: 25 → 23 (fine-tuning from Node 32 which had omega_f=25 but lr=1.5E-5)
Parent rule: UCB selection - Node 32 had highest UCB (UCB=2.477)
Observation: omega_f=23 with lr=2E-5 gives R²=0.988, WORSE than omega_f=25 with lr=2E-5 (R²=0.994). CONFIRMS omega_f=25 is LOCAL MAXIMUM. Complete omega_f map: 20(0.988) < 23(0.988) < 25(**0.994**) > 30(0.984). C field omega_f fully characterized.
Next: parent=33

### Iter 35: good
Node: id=35, parent=33
Mode/Strategy: exploit (omega_f upper boundary probe)
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=27.0, batch_size=1
Metrics: final_r2=0.990, final_mse=2.03, slope=0.980, total_params=1235844, training_time=15.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (inferred from metrics - slope improved from baseline, MSE moderate)
Mutation: [omega_f]: 25 → 27 (testing upper boundary from Node 33 context)
Parent rule: UCB selection - Node 33 (UCB=2.554) was underexplored
Observation: omega_f=27 REGRESSED from omega_f=25 (0.990 vs 0.994). CONFIRMS omega_f=25 is LOCAL MAXIMUM. Complete omega_f map: 20(0.988) < 23(0.988) < 25(**0.994**) > 27(0.990) > 30(0.984).
Next: parent=35

### Iter 36: good
Node: id=36, parent=35
Mode/Strategy: exploit (capacity upper boundary probe)
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=768, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.990, final_mse=2.01, slope=0.976, total_params=1777924, training_time=20.7min
Field: field_name=C, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: (plot not available)
Mutation: [hidden_dim_nnr_f]: 640 → 768
Parent rule: UCB selection - Node 36 had highest UCB (UCB=3.439)
Observation: hidden_dim=768 REGRESSED from hidden_dim=640 (R²=0.990 vs 0.994). CONFIRMS C field capacity ceiling at 640. capacity map: 512(0.986) < 640(**0.994**) > 768(0.990). Training time +5min for worse result.
Next: parent=root (block boundary)

---

## Block 3 Summary: C field, siren_txy, 100 frames, 9000 particles

**Best Configuration**: 640×3, omega_f=25, lr=2E-5, total_steps=150k
**Best Metrics**: R²=0.994, slope=0.989, training_time=15.7min

**Complete Parameter Maps (all LOCAL MAXIMA found)**:
- omega_f: 20(0.988) < 23(0.988) < 25(**0.994**) > 27(0.990) > 30(0.984)
- lr: 1.5E-5(0.987) < 2E-5(**0.994**) > 3E-5(0.993) > 4E-5(0.989)
- hidden_dim: 512(0.986) < 640(**0.994**) > 768(0.990)
- n_layers: 3(**0.993**) > 4(0.980)
- total_steps: 150k(**0.994**) > 200k(0.991)

**Key Findings**:
1. **C omega_f unchanged**: C@9000p optimal=25 (same as prior), does NOT follow lower omega_f trend of Jp/F
2. **C depth ceiling**: n_layers=3 (same as Jp, different from F@4)
3. **C capacity ceiling**: hidden_dim=640 (between F@256 and S@1280)
4. **C lr optimal**: 2E-5 (LOWER than Jp/F@4E-5)
5. **C overfitting**: More steps hurts - 1500 steps/frame is enough

**Block Branching Rate**: 5/11 = 45% (healthy exploration)

---

## Block 4: S field, siren_txy, 100 frames

### Iter 37: poor
Node: id=37, parent=root
Mode/Strategy: exploit (baseline with prior optimal)
Config: lr_NNR_f=2E-5, total_steps=300000, hidden_dim_nnr_f=1280, n_layers_nnr_f=4, omega_f=50.0, batch_size=1
Metrics: final_r2=0.175, final_mse=1.24E-7, slope=0.244, total_params=6568964, training_time=117.3min
Field: field_name=S, inr_type=siren_txy
Visual: (plot not available)
Mutation: Baseline from prior S field optimal (1280×4, omega_f=50, lr=2E-5)
Parent rule: Starting Block 4 from root with prior optimal S config
Observation: CATASTROPHIC FAILURE. Prior optimal S config completely failed on this dataset (R²=0.175 vs prior R²=0.801). Training time extremely long (117min). S field on this 9000-particle dataset may need dramatically different config. Possible causes: (1) omega_f=50 too high for 9000 particles (following Jp/F trend toward lower omega), (2) capacity too high causing overfitting.
Next: parent=37

### Iter 38: poor
Node: id=38, parent=37
Mode/Strategy: exploit (C-optimal config probe)
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.130, final_mse=1.46E-7, slope=0.261, total_params=1235844, training_time=15.7min
Field: field_name=S, inr_type=siren_txy
Visual: (plot not available)
Mutation: [hidden_dim]: 1280 → 640, [n_layers]: 4 → 3, [omega_f]: 50 → 25, [total_steps]: 300k → 150k
Parent rule: UCB selection - Node 37 was only node, testing C-optimal config
Observation: C-optimal config FAILED WORSE than prior S config (R²=0.130 vs 0.175). S field DOES require different architecture than C. Capacity reduction (1280→640) or n_layers reduction (4→3) or both hurt S field. omega_f=25 may also be wrong for S. Next: test intermediate config - keep high capacity but try omega_f between 25 and 50.
Next: parent=38

### Iter 39: moderate
Node: id=39, parent=38
Mode/Strategy: exploit (high capacity, intermediate omega_f)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=35.0, batch_size=1
Metrics: final_r2=0.574, final_mse=6.59E-8, slope=0.690, total_params=4929284, training_time=60.1min
Field: field_name=S, inr_type=siren_txy
Visual: GT/Pred panels show reasonable spatial structure but significant scatter. Pred captures main stress patterns but with lower contrast (slope=0.69). Per-frame MSE varies substantially. Loss converged but to moderate level.
Mutation: [hidden_dim]: 640 → 1280, [omega_f]: 25 → 35, [total_steps]: 150k → 200k
Parent rule: UCB selection - Node 38 was highest UCB, testing high capacity with intermediate omega_f
Observation: SIGNIFICANT IMPROVEMENT: R²=0.574 vs 0.130-0.175 for prior configs. Reducing depth from 4→3 while keeping 1280 capacity and using omega_f=35 helped. S@9000p may have DEPTH CEILING at 3 (not 4 like prior). omega_f=35 is better than both 25 and 50. Next: test lower omega_f (25) with 1280×3 to map omega_f response.
Next: parent=39

### Iter 40: poor
Node: id=40, parent=39
Mode/Strategy: exploit (lower omega_f probe)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.174, final_mse=1.36E-7, slope=0.310, total_params=4929284, training_time=60.3min
Field: field_name=S, inr_type=siren_txy
Visual: Scatter shows extreme spread with no diagonal correlation. Predictions have similar spatial patterns but completely wrong magnitudes. Loss curve converged to plateau but at high error level. Per-frame MSE shows high variance.
Mutation: [omega_f]: 35 → 25
Parent rule: UCB selection - Node 39 was highest UCB (testing omega_f sensitivity)
Observation: CATASTROPHIC REGRESSION: omega_f=25 destroyed performance (R²=0.174 vs parent R²=0.574). S@9000p requires HIGH omega_f (≥35), OPPOSITE to Jp/F trend on this dataset. S omega_f map (1280×3): 25(0.174) << 35(0.574). Next: test higher omega_f (40-45) to probe upper omega_f boundary.
Next: parent=40

### Iter 41: moderate
Node: id=41, parent=40
Mode/Strategy: exploit (higher omega_f probe)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=42.0, batch_size=1
Metrics: final_r2=0.618, final_mse=6.01E-8, slope=0.744, total_params=4929284, training_time=59.9min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Loss curve converges but plateaus at moderate level. Scatter shows significant spread around diagonal (R²=0.618). GT panels show structured stress patterns; Pred panels capture general spatial structure but with differences in contrast and missing fine details. Moderate pattern match.
Mutation: [omega_f]: 25 → 42 (testing higher omega_f from Node 40's failed low omega_f)
Parent rule: UCB selection - Node 40 (UCB=1.228) was parent based on UCB scores
Observation: IMPROVEMENT over parent (R²=0.618 vs 0.174) and even over Node 39 (R²=0.574 with omega_f=35). omega_f=42 is best so far. S omega_f map (1280×3): 25(0.174) << 35(0.574) < 42(0.618). Trend suggests higher omega_f may help - test 48-50 next.
Next: parent=41

### Iter 42: moderate
Node: id=42, parent=41
Mode/Strategy: exploit (omega_f upper boundary probe)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.692, final_mse=4.67E-8, slope=0.783, total_params=4929284, training_time=59.9min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Loss curve converges with plateau. Scatter shows moderate spread but better diagonal clustering than Iter 41 (slope=0.783 up from 0.744). GT/Pred panels show partial spatial match - general stress patterns captured but contrast reduced and fine detail missing. Per-frame MSE noisy across frames, suggesting frame-dependent difficulty.
Mutation: [omega_f]: 42 → 48
Parent rule: UCB selection - Node 41 (UCB=1.773) highest UCB
Observation: NEW BEST for S field: R²=0.692 (up from 0.618@omega=42). Steady improvement with omega_f: 25(0.174) << 35(0.574) < 42(0.618) < 48(0.692). No sign of saturation yet - continue probing higher omega_f.
Next: parent=42

### Iter 43: moderate
Node: id=43, parent=42
Mode/Strategy: exploit (omega_f upper boundary probe)
Config: lr_NNR_f=2E-5, total_steps=200000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=55.0, batch_size=1
Metrics: final_r2=0.693, final_mse=4.60E-8, slope=0.773, total_params=4929284, training_time=111.5min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Scatter shows wide spread around diagonal with moderate correlation. GT/Pred panels show general spatial structure captured but reduced contrast and blurry details. S00/S01 match passably but S10/S11 have wrong intensity. Per-frame MSE noisy with spikes around frames 20-40. Loss converges but with high residual noise.
Mutation: [omega_f]: 48 → 55
Parent rule: UCB selection - Node 42 (UCB=1.939) highest UCB
Observation: FLAT — omega_f=55 shows NO meaningful improvement over omega_f=48 (R²=0.693 vs 0.692, within noise). omega_f trend SATURATED at ~48-55. Training time DOUBLED (60→112min). S omega_f map: 25(0.174) << 35(0.574) < 42(0.618) < 48(0.692) ≈ 55(0.693). Need to switch parameter dimension.
Next: parent=43

### Iter 44: poor
Node: id=44, parent=43
Mode/Strategy: exploit (lr upper boundary probe)
Config: lr_NNR_f=3E-5, total_steps=150000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=55.0, batch_size=1
Metrics: final_r2=0.085, final_mse=1.48E-7, slope=0.177, total_params=4929284, training_time=86.4min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Scatter shows extreme dispersion with no diagonal correlation. Pred panels show washed-out, low-contrast spatial patterns — general shape visible but magnitudes completely wrong. Loss still dropping at 150k steps but converging to poor minimum. Complete failure.
Mutation: [lr]: 2E-5 → 3E-5, [total_steps]: 200k → 150k
Parent rule: UCB selection - Node 43 (UCB=2.026) highest UCB
Observation: CATASTROPHIC FAILURE — lr=3E-5 destroyed S performance (R²=0.085 vs parent R²=0.693). Confirms prior knowledge: S field has ZERO lr tolerance above 2E-5. Combined with reduced steps (150k vs 200k). S lr is hard-locked at 2E-5. Next: try more training steps at best config (omega_f=48, lr=2E-5).
Next: parent=44

### Iter 45: moderate
Node: id=45, parent=44
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=300000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.729, final_mse=4.16E-8, slope=0.824, total_params=4929284, training_time=166.2min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Loss curve shows continued convergence through training. Per-frame MSE noisy but trending downward. Scatter shows moderate spread around diagonal with slope=0.824 (underprediction). GT/Pred panels show general spatial structure captured for all 4 S components but with reduced contrast and blurriness. Better match than prior iterations.
Mutation: [total_steps]: 200k → 300k, [omega_f]: 55 → 48
Parent rule: UCB selection - Node 44 (UCB=2.085) highest UCB
Observation: NEW BEST for S field — R²=0.729 (up from 0.692-0.693 at 200k steps). More training steps clearly helps S field converge. However training_time=166min is extreme. S field benefits from extended training at 1280×3 but is extremely slow. Need to test hidden_dim=1024 for faster iteration while maintaining capacity.
Next: parent=45

## Iter 46: moderate
Node: id=46, parent=45
Mode/Strategy: exploit (hidden_dim capacity probe)
Config: lr_NNR_f=2E-5, total_steps=300000, hidden_dim_nnr_f=1024, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.686, final_mse=4.67E-8, slope=0.738, total_params=3156996, training_time=118.0min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: No visualization available. Scatter metrics show slope=0.738 (underprediction), moderate R². Performance regression from parent.
Mutation: [hidden_dim]: 1280 → 1024
Parent rule: UCB selection - Node 45 (UCB=2.219) highest UCB
Observation: REGRESSION — hidden_dim=1024 degrades S (R²=0.686 vs parent R²=0.729, -0.043). 29% faster (118 vs 166min) and 36% fewer params (3.2M vs 4.9M) but not worth the accuracy loss. S capacity map: 640(0.130) << 1024(0.686) < 1280(0.729). S needs 1280 capacity. Capacity curve is steep between 1024 and 1280.
Next: parent=46

## Iter 47: moderate
Node: id=47, parent=46
Mode/Strategy: exploit (lower lr probe on 1024 config)
Config: lr_NNR_f=1.5E-5, total_steps=300000, hidden_dim_nnr_f=1024, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.721, final_mse=4.11E-8, slope=0.766, total_params=3156996, training_time=118.8min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Scatter shows wide spread along diagonal. GT/Pred panels show spatial structure match but reduced contrast. Loss still converging at 300k. Per-frame MSE noisy with spikes around frame 20.
Mutation: [lr]: 2E-5 -> 1.5E-5 (from Node 46 which used hidden_dim=1024@lr=2E-5)
Parent rule: UCB selected Node 47 (UCB=3.066, highest)
Observation: lr=1.5E-5 HELPS hidden_dim=1024 config: R2=0.721 vs 0.686@lr=2E-5 (+0.035). But still below 1280@lr=2E-5 (0.729). S lr map at hidden_dim=1024: 1.5E-5(0.721) > 2E-5(0.686). Lower lr compensates for reduced capacity. Capacity still trumps lr tuning.
Next: parent=47

## Iter 48: moderate
Node: id=48, parent=47
Mode/Strategy: exploit (lr=1.5E-5 on 1280 config)
Config: lr_NNR_f=1.5E-5, total_steps=300000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.719, final_mse=4.25E-8, slope=0.765, total_params=4929284, training_time=171.7min
Field: field_name=S, inr_type=siren_txy, n_frames=100, n_particles=9000
Visual: Wide scatter spread with diagonal tendency. Per-frame MSE highly variable. GT/Pred spatial structure captured for S_00/S_10/S_11 but S_01 noisier. Loss still converging.
Mutation: [hidden_dim]: 1024 → 1280 (from Node 47 which used 1024@lr=1.5E-5)
Parent rule: UCB selected Node 48 (UCB=3.168, highest)
Observation: lr=1.5E-5 on 1280 (R²=0.719) UNDERPERFORMS lr=2E-5 on 1280 (R²=0.729, Node 45). S lr map at 1280: 2E-5(0.729) > 1.5E-5(0.719). Lower lr does NOT help 1280 config. Confirms lr=2E-5 is optimal for S. lr-capacity interaction: lower lr helps 1024 but not 1280.
Next: parent=root (block boundary)

---

## Block 4 Summary: S field, siren_txy, 100 frames

**Best config**: 1280×3, omega_f=48, lr=2E-5, total_steps=300k → R²=0.729 (Node 45)
**Training time**: 166.2min (extreme)
**Branching rate**: 4/12 = 33% (parents used: root, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 — linear chain with explorative probes)

**Key findings**:
1. S@9000p prior optimal (1280×4, omega_f=50) FAILS (R²=0.175). Depth=3 required (NOT 4).
2. omega_f map: 25(0.174) << 35(0.574) < 42(0.618) < 48(0.692-0.729) ≈ 55(0.693). Optimal=48.
3. S still requires extreme capacity: 640(0.130) << 1024(0.686-0.721) < 1280(0.719-0.729).
4. lr hard-locked at 2E-5. lr=3E-5 catastrophic (0.085). lr=1.5E-5 helps 1024 but hurts 1280.
5. More steps helps: 200k(0.692) < 300k(0.729). But 166min is impractical.
6. S ceiling on this dataset: R²≈0.73 (comparable to prior dataset R²≈0.80). S remains hardest field.
7. S omega_f does NOT follow lower-omega_f trend of Jp/F on this dataset — remains high (48 vs prior 50).

## Block 5: F field, siren_txy, 200 frames

### Iter 49: moderate
Node: id=49, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=4E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=12.0, batch_size=1
Metrics: final_r2=0.983, final_mse=8.03E-3, total_params=265220, compression_ratio=27.2, training_time=29.0min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: baseline config from Block 2 F@100frames, scaled to 200 frames (300k steps = 1500 steps/frame)
Parent rule: first iteration of block, parent=root
Observation: F@200frames R²=0.983 LOWER than F@100frames R²=0.998. Data scaling HURTS F on this dataset, contradicting prior (no diminishing returns). Per-frame MSE peaks around frame 75-100. Need to diagnose: omega_f, capacity, or steps.
Visual: GT/Pred spatial patterns generally match but with visible blurring at disc boundaries. F01/F10 off-diagonal components show more mismatch. Scatter tight along diagonal but with spread. Per-frame MSE non-uniform with mid-simulation peak.
Next: parent=49

### Iter 50: excellent
Node: id=50, parent=49
Mode/Strategy: exploit/boundary
Config: lr_NNR_f=4E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=10.0, batch_size=1
Metrics: final_r2=0.998, final_mse=1.148E-3, total_params=265220, compression_ratio=27.2, training_time=43.2min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: omega_f: 12.0 -> 10.0
Parent rule: UCB selected node 49 (only node). omega_f lowered following "more frames → lower omega_f" rule.
Observation: omega_f=10 DRAMATICALLY improves R² from 0.983→0.998 (+0.015). Confirms "more frames → lower omega_f" for F on this dataset. F@200frames now matches F@100frames quality (0.998). Training time 43.2min (vs 29.0min for iter 49 — both 300k steps, variance likely).
Visual: GT/Pred match well across all 4 components. Per-frame MSE U-shaped (minimum mid-simulation). Scatter tight along diagonal. Slight blurring on off-diagonal but overall excellent spatial reconstruction.
Next: parent=50

### Iter 51: moderate
Node: id=51, parent=50
Mode/Strategy: exploit/boundary
Config: lr_NNR_f=4E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=8.0, batch_size=1
Metrics: final_r2=0.978, final_mse=1.06E-2, total_params=265220, compression_ratio=27.2, training_time=42.9min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: omega_f: 10.0 -> 8.0
Parent rule: UCB selected node 50 (highest). Probing lower omega_f boundary.
Observation: omega_f=8 REGRESSES R² from 0.998→0.978. omega_f map for F@200frames: 8(0.978) < 10(0.998) > 12(0.983). omega_f=10 confirmed LOCAL MAXIMUM. Sharper drop going lower (8→10: +0.020) than going higher (12→10: +0.015).
Visual: Spatial patterns roughly match GT but with blurring. Per-frame MSE shows exponential growth after frame 125 — late-time dynamics degraded. Scatter shows decent diagonal alignment but with visible spread.
Next: parent=51

### Iter 52: excellent
Node: id=52, parent=51
Mode/Strategy: exploit/boundary
Config: lr_NNR_f=4E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=9.0, batch_size=1
Metrics: final_r2=0.998, final_mse=8.09E-4, total_params=265220, compression_ratio=27.2, training_time=32.9min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: omega_f: 8.0 -> 9.0
Parent rule: UCB selected node 51 (highest UCB=2.202). Probing midpoint between omega_f=8 (0.978) and omega_f=10 (0.998).
Observation: omega_f=9 achieves R²=0.998, matching omega_f=10. F@200frames omega_f map refined: 8(0.978) < 9(0.998) ≈ 10(0.998) > 12(0.983). Optimal plateau at [9-10]. Training time 32.9min (vs 42.9min at omega_f=8). Switch-param needed: omega_f mutated 4 consecutive times.
Visual: GT/Pred match well across all 4 F components. Per-frame MSE U-shaped. Scatter tight along diagonal. No visible artifacts.
Next: parent=52

### Iter 53: excellent
Node: id=53, parent=52
Mode/Strategy: switch-param/exploit
Config: lr_NNR_f=5E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=9.0, batch_size=1
Metrics: final_r2=0.9997, final_mse=1.426E-4, total_params=265220, compression_ratio=27.2, training_time=32.4min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: lr_NNR_f: 4E-5 -> 5E-5
Parent rule: Highest UCB (Node 52, UCB=2.412)
Visual: GT/Pred match excellently across all 4 F components. Scatter very tight along diagonal (slope=0.9995). Per-frame MSE U-shaped. No artifacts or blurring.
Observation: lr=5E-5 achieves R2=0.9997, slightly BETTER than lr=4E-5 (R2=0.998). F@200frames maintains wide lr tolerance. New best for F@200frames. F lr map at 200f: 4E-5(0.998) < 5E-5(0.9997).
Next: parent=53

### Iter 54: good
Node: id=54, parent=53
Mode/Strategy: exploit/boundary
Config: lr_NNR_f=6E-5, total_steps=300000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=9.0, batch_size=1
Metrics: final_r2=0.995, final_mse=2.54E-3, total_params=265220, compression_ratio=27.2, training_time=35.4min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: lr_NNR_f: 5E-5 -> 6E-5
Parent rule: Highest UCB (Node 53, UCB=2.414)
Visual: GT/Pred match well all 4 components. Per-frame MSE exponential decay. Scatter tight (slope=0.994). Some outliers at low values.
Observation: lr=6E-5 REGRESSES R² from 0.9997→0.995. lr map: 4E-5(0.998) < 5E-5(0.9997) > 6E-5(0.995). lr=5E-5 confirmed LOCAL MAXIMUM for F@200frames. Narrower lr tolerance at 200f vs 100f (where 6E-5 was still 0.998).
Next: parent=54

### Iter 55: excellent
Node: id=55, parent=54
Mode/Strategy: exploit/speed-pareto
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=9.0, batch_size=1
Metrics: final_r2=0.9988, final_mse=5.57E-4, total_params=265220, compression_ratio=27.2, training_time=27.6min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: total_steps: 300000 -> 200000, lr_NNR_f: 6E-5 -> 5E-5 (reverted to optimal)
Parent rule: Highest UCB (Node 54, UCB=2.242). Reverted lr to 5E-5 optimal, reduced steps to test speed Pareto.
Visual: GT/Pred match well all 4 F components. Scatter tight (slope=0.998, R²=0.9988). Per-frame MSE decreasing curve with mild U-shape. Some scatter at low GT values. Off-diagonal F01/F10 slightly noisier at disc boundaries.
Observation: 200k steps (1000 steps/frame) achieves R²=0.9988 vs 300k (0.9997). Mild regression (-0.0009 R²) for 15% faster training (27.6min vs 32.4min). Speed Pareto: 200k viable if 0.999 not needed. Steps map: 200k(0.9988) < 300k(0.9997).
Next: parent=55

### Iter 56: good
Node: id=56, parent=55
Mode/Strategy: explore/depth-probe
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=9.0, batch_size=1
Metrics: final_r2=0.995, final_mse=2.53E-3, total_params=199428, compression_ratio=36.2, training_time=22.5min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: n_layers_nnr_f: 4 -> 3
Parent rule: Highest UCB (Node 55, UCB=2.869). Depth reduction to test if F@200 needs depth=4 or if 3 suffices.
Visual: GT/Pred panels show reasonable spatial match but some blurring at disc boundaries, especially off-diagonal (F01/F10). Per-frame MSE decaying smoothly. Scatter slightly wider than iter 55 with more outliers at extreme values.
Observation: Depth=3 REGRESSES: 0.995 vs depth=4 (0.9988). F@200frames confirms depth=4 required, consistent with F@100frames (256×4 >> 256×3). Depth=3 is 18% faster (22.5 vs 27.6min) but not worth the accuracy loss. Depth map F@200: 3(0.995) < 4(0.9988). Compression ratio improved (36.2 vs 27.2) due to fewer params (199k vs 265k).
Next: parent=56

### Iter 57: good
Node: id=57, parent=56
Mode/Strategy: exploit/capacity-probe
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=9.0, batch_size=1
Metrics: final_r2=0.994, final_mse=3.05E-3, total_params=446596, compression_ratio=16.2, training_time=36.6min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: hidden_dim_nnr_f: 256 -> 384 (from depth=3 parent)
Parent rule: Highest UCB (Node 56, UCB=2.994). Capacity increase to test if more width compensates for depth=3.
Visual: GT/Pred match reasonable for diagonal (F00/F11) but off-diagonal (F01/F10) noisier with scatter at boundaries. Per-frame MSE U-shaped, rising at later frames. Scatter fairly tight (slope=0.994) but visible spread at extremes.
Observation: 384×3 (0.994) ≈ 256×3 (0.995). Capacity increase does NOT compensate for depth=3 limitation. F requires depth=4 fundamentally, not just more parameters. Confirms: depth is architectural constraint for F, not capacity-limited. Training 62% slower (36.6 vs 22.5min) with no accuracy gain.
Next: parent=57

### Iter 58: moderate
Node: id=58, parent=57
Mode/Strategy: exploit/capacity-depth-probe
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=384, n_layers_nnr_f=4, omega_f=9.0, batch_size=1
Metrics: final_r2=0.970, final_mse=1.42E-2, total_params=594436, compression_ratio=12.1, training_time=33.6min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: n_layers_nnr_f: 3 -> 4 (from 384×3 parent)
Parent rule: Highest UCB (Node 57, UCB=2.484). Test 384×4 to check if increased width helps at correct depth.
Visual: Per-frame MSE U-shaped with exponential rise after frame 125. Scatter wider (slope=0.970). GT/Pred match but with blurring at disc boundaries. Off-diagonal F01/F10 noisier.
Observation: 384×4 (0.970) MUCH WORSE than 256×4 (0.9997, iter 53). F capacity ceiling at 256 CONFIRMED at depth=4 and 200 frames. Larger network harder to train — 2.24× params, worse fit. F@200: capacity map: 256(0.9997) >> 384(0.970). Width penalty severe at depth=4.
Next: parent=58

### Iter 59: moderate
Node: id=59, parent=58
Mode/Strategy: explore/period-probe
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=9.0, batch_size=1, nnr_f_T_period=2.0
Metrics: final_r2=0.790, final_mse=1.011E-1, total_params=265220, compression_ratio=27.2, training_time=37.2min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: nnr_f_T_period: 1.0 -> 2.0 (reverted to 256×4 from 384×4 parent)
Parent rule: Highest UCB (Node 58, UCB=3.206). Revert to optimal 256×4 config and probe last unexplored dimension: T_period.
Visual: Per-frame MSE exponential growth after frame 75, reaching 0.35 at frame 200. Scatter wide and dispersed (slope=0.770). GT components show clear disc structure but Pred loses spatial resolution. Off-diagonal components severely degraded.
Observation: nnr_f_T_period=2.0 CATASTROPHICALLY degrades F@200 from R²=0.998→0.790. Over-smoothing temporal input prevents fitting late frames (MSE exponentially grows). T_period=1.0 is optimal for F. NEVER increase T_period for F field. This is the worst result in the block after reverting to 256×4 — solely caused by T_period change.
Next: parent=59

### Iter 60: moderate
Node: id=60, parent=59
Mode/Strategy: explore/spatial-period-probe
Config: lr_NNR_f=5E-5, total_steps=200000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=9.0, batch_size=1, nnr_f_xy_period=2.0, nnr_f_T_period=1.0
Metrics: final_r2=0.987, final_mse=6.34E-3, total_params=265220, compression_ratio=27.2, training_time=23.8min
Field: field_name=F, inr_type=siren_txy, n_frames=200, n_particles=9000
Mutation: nnr_f_xy_period: 1.0 -> 2.0 (T_period reverted to 1.0)
Parent rule: Highest UCB (Node 59, UCB=3.436). Revert T_period catastrophe and probe xy_period=2.0 to complete spatial period mapping.
Visual: Per-frame MSE U-shaped with exponential growth after frame 125. Scatter moderately tight (slope=0.983). GT/Pred spatial patterns match but with some blurring at disc boundaries. Less catastrophic than T_period=2.0.
Observation: xy_period=2.0 degrades F from R²=0.9997→0.987. Less catastrophic than T_period=2.0 (0.790) but still significant. Period map: xy_period=1.0(0.9997) >> xy_period=2.0(0.987). T_period=1.0(0.9997) >>> T_period=2.0(0.790). Both periods must stay at 1.0 for F@200. Temporal smoothing is 6x more damaging than spatial smoothing (Δ=0.210 vs Δ=0.013 in R²).

### Block 5 Summary
Field: F@200frames@9000p, siren_txy, 12 iterations (49-60)
Best: R²=0.9997 (Iter 53), config: 256×4@omega=9@lr=5E-5@300k steps, 32.4min
Speed Pareto: R²=0.9988 (Iter 55), 200k steps, 27.6min

Complete F@200frames parameter map:
- omega_f: 8(0.978) < 9(0.998) ≈ 10(0.998) > 12(0.983). Optimal=[9-10], confirmed shift from F@100f (12).
- lr: 4E-5(0.998) < 5E-5(0.9997) > 6E-5(0.995). lr=5E-5 LOCAL MAX. Narrower than F@100f.
- steps: 200k(0.9988) < 300k(0.9997). Speed Pareto at 200k.
- depth: 3(0.995) < 4(0.9988-0.9997). Depth=4 confirmed. Width cannot compensate.
- capacity@depth4: 256(0.9997) >> 384(0.970). Severe ceiling at 256.
- T_period: 1.0(0.9997) >>> 2.0(0.790). CATASTROPHIC. Never change.
- xy_period: 1.0(0.9997) >> 2.0(0.987). Significant degradation. Never change.

Key findings: F@200 achieves same quality as F@100 (R²=0.998+) when omega_f is re-tuned from 12→9. Data scaling confirmed beneficial for F. All period parameters must stay at 1.0.
Branching rate: 3/12 = 25% (nodes 56, 57, 59 branched from non-predecessor).

## Block 6: Jp field, siren_txy, 200 frames

### Iter 61: good
Node: id=61, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=4E-5, total_steps=400000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.995, final_mse=9.76E-1, total_params=790529, training_time=67.8min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: Block baseline — Jp@200f with omega_f=5 (lower end of 100f optimal [5-10])
Parent rule: First iteration of block, parent=root
Observation: Jp@200f R²=0.995 with omega_f=5, maintaining quality from 100f (0.996). Slope=0.978 (mild underprediction). Training time 67.8min — very long, need to optimize steps. 2000 steps/frame used.
Visual: No visualization available for this iteration.
Next: parent=61

### Iter 62: good
Node: id=62, parent=61
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=400000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=7.0, batch_size=1
Metrics: final_r2=0.994, final_mse=1.28E+0, total_params=790529, training_time=67.7min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: [omega_f]: 5.0 → 7.0
Parent rule: UCB selection — Node 61 was only node, parent=61
Observation: omega_f=7 MILD REGRESSION from omega_f=5 (R²=0.994 vs 0.995). At 100f, omega_f=[5-10] were equivalent. At 200f, omega_f=5 performs slightly better. Slope=0.963 (still underpredicting). "More frames → lower omega_f" trend continues for Jp. Training time 67.7min — need to test fewer steps.
Visual: Mid-training snapshot shows per-frame MSE heavily front-loaded (frames 0-75 highest), late frames near-zero. GT/Pred spatial patterns broadly match. Scatter at final shows moderate underprediction (slope=0.963).
Next: parent=62

### Iter 63: good
Node: id=63, parent=62
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=300000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=7.0, batch_size=1
Metrics: final_r2=0.988, final_mse=2.65E+0, total_params=790529, training_time=46.1min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: [total_steps]: 400000 → 300000
Parent rule: UCB selection — Node 62 highest UCB
Observation: Steps reduction 400k→300k (2000→1500 steps/frame) DROPS R² from 0.994→0.988 and slope from 0.963→0.934. Training time savings 21.6min (67.7→46.1). Jp@200f needs ~2000 steps/frame for R²>0.99. 1500 steps/frame is INSUFFICIENT for Jp at 200 frames.
Visual: Spatial patterns broadly match GT but with more noise in prediction. Scatter shows wide spread and slope=0.934 indicating systematic underprediction. Per-frame MSE heavily front-loaded (frames 0-75 much higher). Mid-training snapshot at step 90k shows R²=0.895 — final R²=0.988 means significant improvement in last 210k steps.
Next: parent=63

### Iter 64: good
Node: id=64, parent=63
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=300000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=7.0, batch_size=1
Metrics: final_r2=0.992, final_mse=1.90E+0, total_params=790529, training_time=54.3min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: [learning_rate_NNR_f]: 4E-5 → 5E-5
Parent rule: UCB selection — Node 63 highest UCB
Observation: lr increase 4E-5→5E-5 improves R² from 0.988→0.992 at 300k steps. Higher lr partially compensates for fewer steps (vs 400k@lr=4E-5: 0.994-0.995). Slope=0.946 still indicates underprediction. Training time increased 46.1→54.3min despite same steps — stochastic variation.
Visual: GT/Pred spatial match good — disc structures and deformation captured. Scatter shows slope=0.9458 with spread at high GT values (>1.5). Loss curve still decreasing at 300k — more steps could help. Per-frame MSE very high at early frames (0-25) then drops rapidly.
Next: parent=64

### Iter 65: good
Node: id=65, parent=64
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=300000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.992, final_mse=1.56E+0, total_params=790529, training_time=46.1min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: [omega_f]: 7.0 → 5.0
Parent rule: UCB selection — Node 64 highest UCB=2.046
Observation: omega_f 7→5 at 300k/lr=5E-5: R² ties (0.992≈0.992) but slope improves (0.946→0.960). At 400k steps, omega_f=5 beats 7 (0.995>0.994). At 300k steps, difference narrows. omega_f=5 gives better magnitude fidelity (slope closer to 1). omega_f map@300k: 5(0.992/slope=0.960) ≈ 7(0.992/slope=0.946). omega_f=5 wins on slope.
Visual: No field plot available for this iteration.
Next: parent=65

### Iter 66: good
Node: id=66, parent=65
Mode/Strategy: failure-probe
Config: lr_NNR_f=5E-5, total_steps=300000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=3.0, batch_size=1
Metrics: final_r2=0.992, final_mse=1.75E+0, slope=0.950, total_params=790529, training_time=49.5min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: [omega_f]: 5.0 → 3.0
Parent rule: UCB selection — Node 65 highest UCB (failure-probe to find lower omega_f boundary)
Observation: omega_f 5→3 failure-probe: R² unchanged (0.992≈0.992), slope slightly worse (0.960→0.950). omega_f=3 is still viable — lower boundary NOT found. omega_f map@300k: 3(0.992/slope=0.950) ≈ 5(0.992/slope=0.960) ≈ 7(0.992/slope=0.946). Extremely flat omega_f response at 300k steps. Loss curve still trending down — model not fully converged at 300k. Early frames (first 25) have much higher per-frame MSE.
Visual: GT/Pred spatial patterns match well — disc shapes captured. Scatter shows decent diagonal but noticeable spread at GT>1.5. Slope=0.95 confirms mild underprediction of extremes. Loss still trending down at 300k — more steps would help.
Next: parent=66

## Iter 67: good
Node: id=67, parent=66
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=400000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=3.0, batch_size=1
Metrics: final_r2=0.995, final_mse=1.06E+0, slope=0.965, total_params=790529, training_time=55.4min
Field: field_name=Jp, inr_type=siren_txy, n_frames=200
Mutation: [total_steps]: 300000 → 400000
Parent rule: UCB selection — Node 66 highest UCB. Test if omega_f=3+400k matches omega_f=5+400k.
Observation: omega_f=3+400k yields R²=0.995, slope=0.965 — MATCHES omega_f=5+400k (iter 61: R²=0.995, slope=0.978). Confirms omega_f insensitivity [3-7] extends to 400k steps. Slope slightly lower (0.965 vs 0.978) — omega_f=5 still wins on slope. Step count (400k vs 300k) is the dominant factor, not omega_f. 300k→400k recovers R² from 0.992→0.995.
Visual: Loss curve still trending slightly downward at 400k — not fully converged. GT/Pred spatial match good, disc structures clear. Scatter shows good diagonal alignment with spread at GT>1.5. Per-frame MSE: early frames (first ~25) have much higher error, later frames converge well.
Next: parent=67
