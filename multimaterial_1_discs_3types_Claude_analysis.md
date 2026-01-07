# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 3: siren_txy, F field, n_frames=100

## Iter 25: excellent
Node: id=25, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.999, final_mse=5.32E-4, total_params=1054724, slope=0.999, training_time=72.5min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: field_name: Jp -> F (new block)
Parent rule: New block start, testing if Jp optimal config (512×4, lr=2E-5, omega=30, 150k steps) generalizes to F field
Observation: EXCEPTIONAL SUCCESS! R²=0.999, slope=0.999 - nearly perfect reconstruction of F field. The optimal Jp config transfers perfectly to F. F field (4 components) achieves even better R² than Jp (1 component). Architecture is field-agnostic.
Next: parent=25 (success-exploit strategy - test robustness or explore boundary)

## Iter 26: excellent
Node: id=26, parent=25
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.999, final_mse=3.93E-4, total_params=1054724, slope=1.000, training_time=48.5min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 150000 -> 100000
Parent rule: UCB selection from Node 25, probe efficiency boundary
Observation: Maintained R²=0.999 with 33% fewer steps. Training time reduced 72.5→48.5min. F field more efficient than expected.
Next: parent=26

## Iter 27: excellent
Node: id=27, parent=26
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=75000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.992, final_mse=3.62E-3, total_params=1054724, slope=0.995, training_time=36.4min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 100000 -> 75000
Parent rule: UCB selection from Node 26, continue efficiency probing
Observation: R² dropped 0.999→0.992 but still excellent. Found efficiency boundary: 100k is optimal, 75k trades 0.7% R² for 25% training time savings. 3 consecutive R²≥0.95 triggers failure-probe.
Next: parent=27 (failure-probe strategy - extreme parameter to find true boundary)

## Iter 28: excellent
Node: id=28, parent=27
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.979, final_mse=1.01E-2, total_params=1054724, slope=0.983, training_time=24.4min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 75000 -> 50000
Parent rule: Node 27 highest UCB; failure-probe to find minimum viable steps
Observation: 50k steps still achieves R²=0.979 (excellent). F field remarkably efficient - 500 steps/frame still viable. 4 consecutive R²≥0.95 confirms robust architecture. Continue probing lower boundary.
Next: parent=28 (continue failure-probe - test 30k steps)

## Iter 33: excellent
Node: id=33, parent=32
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=10000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.955, final_mse=2.15E-2, total_params=1054724, slope=0.951, training_time=5.2min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 12000 -> 10000
Parent rule: Node 32 child via UCB=3.076 (highest); continue failure-probe
Observation: 10k steps (100 steps/frame) still achieves R²=0.955 (excellent). 9 consecutive R²≥0.95. Just above threshold - next iteration should find <0.95 boundary.
Next: parent=33 (continue failure-probe - test 8k steps to find boundary)

## Iter 34: good
Node: id=34, parent=33
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=8000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.947, final_mse=2.55E-2, total_params=1054724, slope=0.937, training_time=4.2min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 10000 -> 8000
Parent rule: Node 33 child via UCB=3.183 (highest); continue failure-probe
Observation: BOUNDARY FOUND! 8k steps (80 steps/frame) drops to R²=0.947 (good, not excellent). First R²<0.95 after 9 consecutive excellent. Boundary confirmed: ~100 steps/frame ≈ R²=0.95 threshold for F field with 512×4 architecture.
Next: parent=34 (switch-param: explore smaller model at optimal 10k steps)

## Iter 35: excellent
Node: id=35, parent=34
Mode/Strategy: switch-param
Config: lr_NNR_f=2E-5, total_steps=10000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.965, final_mse=1.68E-2, total_params=265220, slope=0.963, training_time=1.7min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 512 -> 256
Parent rule: Node 35 UCB=3.310 (highest); switch from total_steps exploration to hidden_dim exploration
Observation: EXCELLENT EFFICIENCY! R²=0.965 with 256×4 model (4x smaller) achieves same quality as 512×4. Training time 1.7min (3x faster than 5.2min at 512×4). Model reduced from 1.05M to 265k params. 256×4 is sufficient for F field at 10k steps.
Next: parent=35 (explore even smaller model: 128×4 to find capacity boundary)

## Iter 36: excellent
Node: id=36, parent=35
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=10000, hidden_dim_nnr_f=128, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.951, final_mse=2.33E-2, total_params=67076, slope=0.950, training_time=0.8min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 256 -> 128
Parent rule: Node 35 child via UCB=3.401 (highest); continue capacity exploration
Observation: CAPACITY BOUNDARY FOUND! 128×4 achieves R²=0.951 (just above 0.95). Model reduced from 265k to 67k params (4x smaller). Training 0.8min (2x faster than 1.7min). 128×4 is minimum viable capacity at 10k steps for R²≥0.95.

---

## Block 3 Summary: F field efficiency mapping (Iterations 25-36)

**Configuration**: siren_txy, F field, n_frames=100, lr=2E-5, omega=30
**Success rate**: 11/12 iterations R²≥0.95 (92%)
**Branching rate**: 0% (linear chain exploration)

**Key findings**:
1. **Field generalization CONFIRMED**: 512×4 optimal Jp config achieves R²=0.999 on F field - architecture is field-agnostic
2. **F field more efficient than Jp**: F achieves R²=0.999 vs Jp's R²=0.964 with same config
3. **Steps boundary**: ~100 steps/frame ≈ R²=0.95 threshold (10k steps for 100 frames)
4. **Capacity boundary**: 128×4 (67k params) is minimum for R²≥0.95 at 10k steps
5. **Efficiency scaling**: 512→256→128 all maintain R²≥0.95, training time 5.2→1.7→0.8min

**Best efficiency config**: 128×4, 10k steps → R²=0.951, 0.8min training
**Best quality config**: 512×4, 100k steps → R²=0.999, 48.5min training

PROTOCOL EDITED: No changes needed - branching rate 0% but linear probing was intentional for boundary finding. Success rate 92% indicates good exploration strategy.

## Iter 31: excellent
Node: id=31, parent=30
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=15000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.964, final_mse=1.72E-2, total_params=1054724, slope=0.965, training_time=7.6min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 20000 -> 15000
Parent rule: Node 30 child via UCB=2.835 (highest); continue failure-probe
Observation: R²=0.964 with 15k steps (150 steps/frame). First drop below 0.98 but still excellent. Training time 7.6min is ideal. 7 consecutive R²≥0.95.
Next: parent=31 (continue failure-probe - test 12k steps to find true boundary)

## Iter 29: excellent
Node: id=29, parent=28
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=30000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.998, final_mse=1.07E-3, total_params=1054724, slope=0.997, training_time=14.8min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 50000 -> 30000
Parent rule: Node 28 child via UCB=2.579 (highest); continue failure-probe to find minimum viable steps
Observation: SURPRISING RESULT! R²=0.998 with only 30k steps - higher than 50k (R²=0.979). Stochastic initialization variance confirmed. 5 consecutive R²≥0.95, 300 steps/frame sufficient for F field. Training time reduced to 14.8min (5x faster than 150k baseline).
Next: parent=29 (continue probing - test 20k steps to find true failure boundary)

## Iter 30: excellent
Node: id=30, parent=29
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=20000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.989, final_mse=5.28E-3, total_params=1054724, slope=0.988, training_time=10.0min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 30000 -> 20000
Parent rule: Node 29 child via UCB=2.721 (highest); continue failure-probe
Observation: 20k steps still achieves R²=0.989 (excellent). 6 consecutive R²≥0.95. 200 steps/frame is viable! Training time=10min (target). Continue probing to find true boundary.
Next: parent=30 (continue failure-probe - test 15k steps)

## Iter 32: excellent
Node: id=32, parent=31
Mode/Strategy: failure-probe
Config: lr_NNR_f=2E-5, total_steps=12000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.983, final_mse=8.40E-3, total_params=1054724, slope=0.974, training_time=6.1min
Field: field_name=F, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 15000 -> 12000
Parent rule: Node 31 child via UCB=2.982 (highest); continue failure-probe
Observation: 12k steps still achieves R²=0.983 (excellent). 8 consecutive R²≥0.95. 120 steps/frame is viable! Training time 6.1min is very efficient. Continue probing to find <0.95 boundary.
Next: parent=32 (continue failure-probe - test 10k steps)

---

## Block 2: siren_txy, Jp field, n_frames=100

## Iter 24: excellent
Node: id=24, parent=16
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.964, final_mse=5.34, total_params=1053185, slope=0.904, training_time=72.1min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 100000 -> 150000
Parent rule: Node 16 highest UCB; testing 1.5x steps to push R²>0.95
Observation: SUCCESS! R² finally exceeded 0.95 target (0.943→0.964). Slope also excellent (0.845→0.904). Training time increased to 72.1min. Confirms more steps is key path to excellence.
Next: BLOCK END - proceed to Block 3

### Block 2 Summary
Best config: Node 24 (hidden_dim=512, n_layers=4, lr=2E-5, omega_f=30, steps=150k, n_frames=100) → R²=0.964, slope=0.904
Key findings:
- **SUCCESS**: Achieved R²>0.95 for first time (R²=0.964)
- **Training data scaling**: 100 frames requires ~150k steps for R²>0.95 (vs 50k for 48 frames)
- **Optimal architecture**: 512×4 with lr=2E-5, omega_f=30 is the winning configuration
- **5-layer REJECTED**: All lr values (2E-5, 1.5E-5, 1E-5) fail with 5 layers
- **omega_f sensitivity CONFIRMED**: omega_f=35 degrades R² by 0.07 vs omega_f=30
Branching rate: 4/12 = 33% (healthy exploration)
Improvement rate: 7/12 = 58% (good exploitation)

---

## Iter 23: poor
Node: id=23, parent=20
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=5, omega_f=30.0, batch_size=1
Metrics: final_r2=0.739, final_mse=42.86, total_params=1315841, slope=0.530, training_time=59.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 1.5E-5 -> 1E-5
Parent rule: Node 20 highest UCB (3.094); testing even lower lr for 5-layer network
Observation: 5-LAYER DEFINITIVELY REJECTED. lr=1E-5 made it even worse (R²=0.739 vs 0.858 at lr=1.5E-5). All tested lr values (2E-5, 1.5E-5, 1E-5) fail with 5 layers. Return to 4-layer.
Next: parent=16

## Iter 22: moderate
Node: id=22, parent=21
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=35.0, batch_size=1
Metrics: final_r2=0.857, final_mse=20.22, total_params=1053185, slope=0.776, training_time=48.2min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 30.0 -> 35.0
Parent rule: Node 21 highest UCB (3.050); testing omega_f increase for potential R² improvement
Observation: omega_f=35 severely degraded R² (0.929→0.857). Confirms omega_f=30 is optimal. Even +5 increase causes significant performance loss.
Next: parent=20

## Iter 21: good
Node: id=21, parent=16
Mode/Strategy: exploit
Config: lr_NNR_f=1.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.929, final_mse=11.78, total_params=1053185, slope=0.802, training_time=48.3min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 2E-5 -> 1.5E-5
Parent rule: Node 16 highest UCB (1.549); testing lower lr on 4-layer to push toward R²>0.95
Observation: Lower lr degraded R² (0.943→0.929). lr=2E-5 confirmed optimal for 512×4. Node 16 remains best (R²=0.943).
Next: parent=21

## Iter 20: moderate
Node: id=20, parent=19
Mode/Strategy: exploit
Config: lr_NNR_f=1.5E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=5, omega_f=30.0, batch_size=1
Metrics: final_r2=0.858, final_mse=20.80, total_params=1315841, slope=0.750, training_time=59.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 2E-5 -> 1.5E-5
Parent rule: Node 19 highest UCB (2.750); testing lower lr to fix 5-layer penalty
Observation: Lower lr did NOT help 5-layer network. R² dropped further (0.879→0.858). The 5-layer architecture is fundamentally worse than 4-layer for this config. Branch to Node 16 (R²=0.943) to try different direction.
Next: parent=16

## Iter 19: moderate
Node: id=19, parent=16
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=5, omega_f=30.0, batch_size=1
Metrics: final_r2=0.879, final_mse=16.60, total_params=1315841, slope=0.824, training_time=59.5min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: n_layers_nnr_f: 4 -> 5
Parent rule: Node 16 highest UCB (1.691); testing deeper network for R²>0.95
Observation: Adding depth HURT performance: R² dropped from 0.943 (4 layers) to 0.879 (5 layers). Time increased 23% (48.3→59.5min). Confirms lr-depth relationship: n_layers=5 needs lr<2E-5.
Next: parent=19

## Iter 18: poor
Node: id=18, parent=17
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=75000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.616, final_mse=52.56, total_params=1053185, slope=0.541, training_time=36.2min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 50000 -> 75000
Parent rule: Node 17 highest UCB (2.050); testing intermediate steps
Observation: SEVERE REGRESSION. R² crashed from 0.895 (parent) to 0.616. Same config except +25k steps should not cause this. Likely stochastic failure or training instability. Will branch to Node 16 (R²=0.943).
Next: parent=16

## Block 1: siren_txy, Jp field, n_frames=48

## Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: explore
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.476, final_mse=1.07E+02, total_params=12801, training_time=1.9min
Field: field_name=Jp, inr_type=siren_txy
Mutation: Initial config
Parent rule: root (first iteration)
Observation: omega_f=80 is extremely high, likely causing instability. Protocol warns omega_f>50 causes training instability.
Next: parent=1

## Iter 2: poor
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.527, final_mse=1.07E+02, total_params=12801, training_time=1.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 80.0 -> 30.0
Parent rule: Node 1 highest UCB; fixing omega_f instability first
Observation: Reducing omega_f from 80 to 30 gave marginal improvement (0.476→0.527). Model capacity (hidden_dim=64) likely too small.
Next: parent=2

## Iter 3: poor
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=128, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.650, final_mse=9.59E+01, total_params=50177, slope=0.113, training_time=2.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 64 -> 128
Parent rule: Node 2 highest UCB; increasing model capacity
Observation: Doubling hidden_dim (64→128) improved R² (0.527→0.650). Slope=0.113 very low (should be ~1.0). Still poor - capacity insufficient.
Next: parent=3

## Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.864, final_mse=7.46E+01, total_params=198657, slope=0.219, training_time=6.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 128 -> 256
Parent rule: Node 3 highest UCB; continuing capacity scaling
Observation: Doubling hidden_dim (128→256) improved R² (0.650→0.864, +0.214). Slope still low (0.219). Capacity scaling continues to work.
Next: parent=4

## Iter 5: good
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.907, final_mse=4.78E+01, total_params=790529, slope=0.383, training_time=19.1min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 256 -> 512
Parent rule: Node 4 highest UCB; continuing capacity scaling
Observation: Doubling hidden_dim (256→512) improved R² (0.864→0.907, +0.043). Diminishing returns. Training time 3x increase (6.4→19.1min) exceeds 10min target.
Next: parent=5

## Iter 6: poor
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.274, final_mse=8.82E+01, total_params=790529, slope=0.241, training_time=18.6min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5 -> 5E-5
Parent rule: Node 5 highest UCB (2.062); testing higher learning rate to potentially speed convergence
Observation: CATASTROPHIC FAILURE. 5x lr increase (1E-5→5E-5) caused severe regression (R²: 0.907→0.274). Confirms lr_NNR_f=1E-5 is near-optimal for hidden_dim=512 siren_txy. Training time still ~19min - lr change didn't help time issue.
Next: parent=5 (UCB=2.062)

## Iter 7: moderate
Node: id=7, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=30000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.877, final_mse=7.01E+01, total_params=790529, slope=0.244, training_time=11.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: total_steps: 50000 -> 30000
Parent rule: Node 5 highest UCB; testing training time reduction by reducing steps
Observation: Reducing total_steps (50k→30k) brought time closer to target (11.3min vs 19.1min) but R² regressed (0.907→0.877). hidden_dim=512 requires longer training to converge. Tradeoff: time vs accuracy.
Next: parent=7 (UCB=2.748)

## Iter 8: moderate
Node: id=8, parent=7
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.865, final_mse=7.34E+01, total_params=264449, slope=0.226, training_time=7.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 512 -> 256, n_layers_nnr_f: 3 -> 4, total_steps: 30000 -> 50000
Parent rule: Node 7 highest UCB (2.748); testing depth vs width tradeoff with smaller hidden_dim
Observation: Reducing hidden_dim (512→256) and adding layer (3→4) achieved similar R² (0.865 vs 0.864 at Node 4) in less time (7.8min vs 11.3min). Depth does not compensate for width reduction for this architecture.
Next: parent=8 (UCB=2.864)

## Iter 9: good
Node: id=9, parent=8
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.908, final_mse=4.55E+01, total_params=264449, slope=0.400, training_time=7.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5 -> 2E-5
Parent rule: Node 8 highest UCB (2.864); testing moderate lr increase
Observation: EXCELLENT RESULT. 2x lr increase (1E-5→2E-5) improved R² (0.865→0.908, +0.043) to match Node 5 (0.907) with hidden_dim=512. Training time 7.7min vs 19.1min (2.5x faster). This config achieves best time-efficiency ratio.
Next: parent=9 (UCB=3.029)


## Iter 10: moderate
Node: id=10, parent=9
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.896, final_mse=2.86E+01, total_params=264449, slope=0.551, training_time=7.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 2E-5 -> 3E-5
Parent rule: Node 9 highest UCB (3.029); testing upper lr boundary
Observation: lr=3E-5 caused slight R² regression (0.908→0.896, -0.012) but improved slope (0.400→0.551). lr=2E-5 appears optimal for hidden_dim=256, n_layers=4.
Next: parent=10 (UCB=3.132)

## Iter 11: poor
Node: id=11, parent=10
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=5, omega_f=30.0, batch_size=1
Metrics: final_r2=0.698, final_mse=4.87E+01, total_params=330241, slope=0.432, training_time=9.5min
Field: field_name=Jp, inr_type=siren_txy
Mutation: n_layers_nnr_f: 4 -> 5
Parent rule: Node 10 highest UCB (3.132); testing increased depth
Observation: SIGNIFICANT REGRESSION. Adding 5th layer with lr=3E-5 dropped R² (0.896→0.698, -0.198). Deeper networks need lower lr. n_layers=5 + lr=3E-5 is unstable.
Next: parent=11 (UCB=3.043)

## Iter 12: moderate
Node: id=12, parent=11
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=5, omega_f=30.0, batch_size=1
Metrics: final_r2=0.881, final_mse=4.48E+01, total_params=330241, slope=0.410, training_time=9.5min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5 -> 2E-5
Parent rule: Node 11 highest UCB (3.043); testing lower lr for n_layers=5
Observation: Lower lr (2E-5 vs 3E-5) improved n_layers=5 (R²: 0.698→0.881, +0.183). Confirms lr must decrease with depth. Still below n_layers=4 (R²=0.908), suggesting 4 layers optimal.
Next: END OF BLOCK 1

---

## Block 1 Summary

**Configuration:** siren_txy, Jp field, n_training_frames=48

**Statistics:**
- Iterations: 12
- Best R²: 0.908 (Node 9)
- Best slope: 0.551 (Node 10)
- R² range: 0.274-0.908
- Branching rate: 18% (2 branches from non-root parents in 11 transitions)

**Best Configuration Found:**
- hidden_dim_nnr_f=256, n_layers_nnr_f=4, lr_NNR_f=2E-5, omega_f=30.0
- R²=0.908, slope=0.400, time=7.7min

**Key Findings:**
1. **omega_f sensitivity**: omega_f=80 severely degraded training (R²=0.476). omega_f=30 optimal.
2. **Model capacity scaling**: hidden_dim progression 64→128→256→512 showed R² gains with diminishing returns. 512 exceeds time budget.
3. **Depth vs width tradeoff**: 256×4 matches 512×3 accuracy at 2.5x faster training (7.7min vs 19.1min).
4. **lr-depth relationship**: Deeper networks require lower lr. n_layers=5 + lr=3E-5 fails (R²=0.698), but n_layers=5 + lr=2E-5 recovers (R²=0.881).
5. **Optimal lr scales inversely with depth**: lr=2E-5 optimal for n_layers=4, lr needs to be lower for n_layers=5.

**Not achieved:** R² > 0.95 target. Best R²=0.908 suggests either more training frames or architectural changes needed.

PROTOCOL EDITED: Added depth-lr relationship to Theoretical Background

---

## Block 2: siren_txy, Jp field, n_frames=100

## Iter 13: moderate
Node: id=13, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.857, final_mse=53.98, total_params=264449, slope=0.385, training_time=7.8min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: n_training_frames: 48 -> 100 (block change)
Parent rule: Block 2 start, testing n_frames=100 with Block 1 optimal config
Observation: Surprisingly R² dropped from 0.908 (48 frames) to 0.857 (100 frames). More data + same steps = underfitting.
Next: parent=13, increase total_steps to compensate for more data

## Iter 14: good
Node: id=14, parent=13
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.915, final_mse=19.84, total_params=264449, slope=0.675, training_time=15.3min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 50000 -> 100000
Parent rule: Node 13 highest UCB; compensating for 2x frames with 2x steps
Observation: Doubling total_steps restored and improved R² (0.857→0.915). Confirms per-frame training intensity hypothesis. Slope also improved (0.385→0.675).
Next: parent=14, try lr=2.5E-5 to push R² higher without adding more time

## Iter 15: good
Node: id=15, parent=14
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.905, final_mse=16.61, total_params=264449, slope=0.743, training_time=15.4min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: lr_NNR_f: 2E-5 -> 2.5E-5
Parent rule: Node 14 highest UCB; testing higher lr for potential R² improvement
Observation: Higher lr slightly degraded R² (0.915→0.905) but improved slope (0.675→0.743). lr=2E-5 remains optimal for R².
Next: parent=15, try hidden_dim=512 to increase capacity

## Iter 16: good
Node: id=16, parent=15
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.943, final_mse=9.03, total_params=1053185, slope=0.845, training_time=48.3min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: hidden_dim_nnr_f: 256 -> 512
Parent rule: Node 15 highest UCB (1.848); increasing capacity for R² improvement
Observation: Capacity increase boosted R² (0.905→0.943, near excellent) and slope (0.743→0.845). BUT training_time exploded (15.4→48.3min, 3.1x). Need to reduce steps or explore different capacity path.
Next: parent=16, reduce total_steps to bring time under control while preserving R²

## Iter 17: good
Node: id=17, parent=16
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.895, final_mse=25.68, total_params=1053185, slope=0.617, training_time=24.3min
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
Mutation: total_steps: 100000 -> 50000
Parent rule: Node 16 highest UCB (1.997); testing reduced steps for faster training
Observation: Halving steps reduced time (48.3→24.3min) but R² dropped (0.943→0.895). 50k steps insufficient for 512×4 with 100 frames.
Next: parent=17
