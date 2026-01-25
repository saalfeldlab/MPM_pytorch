# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|

### Established Principles
(From prior knowledge - to be validated in this experiment)
- Jp field optimal config: hidden_dim=384, n_layers=3, omega_f=15-20 (200f) or 30-35 (48f)
- For 100 frames: expect omega_f=20-25 to be optimal
- Steps/frame ~1000 needed for R²>0.99
- Batch_size=1 required for training time efficiency

### Open Questions
- Does the multimaterial_1_discs_3types dataset behave similarly to prior experiments?
- What are the optimal hyperparameters for this specific dataset?

---

## Previous Block Summary

(No previous block)

---

## Current Block (Block 1)

### Block Info
Field: field_name=Jp, inr_type=siren_txy
n_training_frames: 100
Iterations: 1-12

### Hypothesis
Starting with severely underfit baseline (hidden_dim=64, omega_f=80). Expect dramatic improvement by moving towards known Jp optimal config (hidden_dim=384, omega_f=20-25, lr=4E-5).

### Iterations This Block

#### Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: explore (initial baseline)
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.416, final_mse=1.22E+02, total_params=12801, slope=0.049, training_time=2.4min
Mutation: Initial config (baseline)
Observation: SEVERE UNDERFIT - hidden_dim=64 too small, omega_f=80 too high for Jp field
Next: parent=1, increase hidden_dim to 256 and reduce omega_f to 25

#### Iter 2: moderate
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.799, final_mse=8.92E+01, total_params=198657, slope=0.189, training_time=2.9min
Mutation: hidden_dim: 64 -> 256
Observation: +0.383 R² from capacity increase. omega_f=80 still too high (4x optimal). Slope=0.189 = severe underprediction.
Next: parent=2, reduce omega_f from 80 to 25

#### Iter 3: moderate
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.774, final_mse=9.16E+01, total_params=198657, slope=0.178, training_time=2.7min
Mutation: omega_f: 80 -> 25
Visual: SSIM=0.998 excellent spatial match, scatter shows severe underprediction
Observation: omega_f 80→25 unexpectedly WORSE (-0.025 R²). lr_NNR_f=1E-5 is likely bottleneck.
Next: parent=3 (highest UCB), increase lr_NNR_f from 1E-5 to 3E-5

#### Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.873, final_mse=4.38E+01, total_params=198657, slope=0.455, training_time=2.5min
Mutation: lr_NNR_f: 1E-5 -> 3E-5
Observation: LR increase confirmed as bottleneck! +0.099 R², slope 2.6x better. Need more capacity.
Next: parent=4 (highest UCB=2.287), increase hidden_dim from 256 to 384

#### Iter 5: moderate
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.867, final_mse=2.62E+01, total_params=445441, slope=0.634, training_time=3.3min
Mutation: hidden_dim: 256 -> 384
Observation: Unexpected - R² dropped -0.006 but slope improved +0.179 (0.455→0.634). More capacity helps slope but 50k steps insufficient for 384-dim model. Steps/frame=500, need 800+.
Next: parent=5 (highest UCB=2.448), increase total_steps from 50000 to 80000

#### Iter 6: good
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=80000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.937, final_mse=1.06E+01, total_params=445441, slope=0.813, training_time=5.0min
Mutation: total_steps: 50000 -> 80000
Observation: +0.070 R², +0.179 slope! Steps increase confirmed bottleneck. Now approaching excellent (0.95 threshold). 800 steps/frame working.
Next: parent=6 (highest UCB=2.669), increase lr_NNR_f from 3E-5 to 4E-5 (prior optimal for Jp)

#### Iter 7: excellent
Node: id=7, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=80000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1
Metrics: final_r2=0.960, final_mse=7.08E+00, total_params=445441, slope=0.848, training_time=5.0min
Mutation: lr_NNR_f: 3E-5 -> 4E-5
Visual: SSIM=1.000 perfect spatial match. Scatter shows tight cloud along diagonal with slope=0.848.
Observation: EXCELLENT achieved! lr_NNR_f=4E-5 confirmed as optimal for Jp. +0.023 R², +0.035 slope.
Next: parent=7 (highest UCB=2.831), reduce omega_f from 25 to 20 to test if slope improves

### Emerging Observations
- **Trajectory: poor(0.416) → moderate(0.799) → moderate(0.774) → moderate(0.873) → moderate(0.867) → good(0.937) → EXCELLENT(0.960)**
- hidden_dim 64→256 gave +0.383 R² (capacity was bottleneck initially)
- omega_f 80→25 gave -0.025 R² (NOT the bottleneck at low lr)
- lr_NNR_f 1E-5→3E-5 gave +0.099 R² (CONFIRMED bottleneck)
- hidden_dim 256→384 gave -0.006 R² but +0.179 slope (MORE STEPS NEEDED for larger model)
- total_steps 50k→80k gave +0.070 R², +0.179 slope (STEPS INCREASE CONFIRMED)
- lr_NNR_f 3E-5→4E-5 gave +0.023 R², +0.035 slope (LR OPTIMAL AT 4E-5 for Jp)
- Slope improved steadily: 0.049 → 0.189 → 0.178 → 0.455 → 0.634 → 0.813 → 0.848 (approaching 1.0!)
- **Current state**: R²=0.960 (EXCELLENT!), slope=0.848, SSIM=1.000 perfect spatial match
- **Optimal config so far**: lr=4E-5, hidden_dim=384, n_layers=3, omega_f=25, 80k steps
- **Next test**: omega_f=20 (prior suggests 20-25 optimal for Jp@100frames, may improve slope)
