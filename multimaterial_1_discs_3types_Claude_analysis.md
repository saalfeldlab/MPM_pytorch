# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1: Jp field exploration with siren_txy (100 frames)

### Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: explore (initial baseline)
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.416, final_mse=1.22E+02, total_params=12801, slope=0.049, training_time=2.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: Initial config (baseline)
Parent rule: root - starting exploration
Observation: SEVERE UNDERFIT - hidden_dim=64 is 6x smaller than known Jp optimal (384), omega_f=80 is 4x higher than optimal (20-25). Model lacks capacity and has wrong frequency scale.
Next: parent=1, dramatically increase hidden_dim and reduce omega_f

### Iter 2: moderate
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=80.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.799, final_mse=8.92E+01, total_params=198657, slope=0.189, training_time=2.9min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim: 64 -> 256
Parent rule: Continue from poor baseline, increase capacity
Observation: +0.383 R² from capacity increase alone (64→256). Still underfit - omega_f=80 is 4x too high for Jp@100frames. Slope=0.189 indicates severe underprediction.
Next: parent=2, reduce omega_f from 80 to 25 (near known optimal)

### Iter 3: moderate
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=25.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.774, final_mse=9.16E+01, total_params=198657, slope=0.178, training_time=2.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 80 -> 25
Visual: SSIM=0.998 excellent spatial match. Scatter shows severe underprediction - pred clustered around 1.0 while GT spans 0.5-2.0. Loss still declining at 50k steps.
Parent rule: UCB selection - node 2 had highest UCB
Observation: omega_f reduction 80→25 SLIGHTLY WORSE (-0.025 R²). Unexpected - suggests lr_NNR_f=1E-5 is bottleneck, not omega_f. Need to increase lr.
Next: parent=3 (highest UCB=1.999), increase lr_NNR_f from 1E-5 to 3E-5

### Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=25.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.873, final_mse=4.38E+01, total_params=198657, slope=0.455, training_time=2.5min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5 -> 3E-5
Parent rule: UCB selection - node 3 had highest UCB
Observation: LR increase confirmed as bottleneck! +0.099 R², slope 0.178→0.455 (2.6x). Now model captures more dynamic range. Still underfit - need more capacity (hidden_dim=384 is Jp optimal).
Next: parent=4 (highest UCB=2.287), increase hidden_dim from 256 to 384

### Iter 5: moderate
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.867, final_mse=2.62E+01, total_params=445441, slope=0.634, training_time=3.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim: 256 -> 384
Parent rule: UCB selection - node 4 had highest UCB=2.287
Observation: Unexpected tradeoff - R² dropped -0.006 (0.873→0.867) but slope improved +0.179 (0.455→0.634). Larger model captures dynamic range better but needs more training. 50k steps = 500 steps/frame insufficient for 445k param model.
Next: parent=5 (highest UCB=2.448), increase total_steps from 50000 to 80000 (800 steps/frame)

### Iter 6: good
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=80000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.937, final_mse=1.06E+01, total_params=445441, slope=0.813, training_time=5.0min
Field: field_name=Jp, inr_type=siren_txy
Mutation: total_steps: 50000 -> 80000
Parent rule: Exploit highest UCB node (Node 5), test hypothesis that larger model needs more steps
Observation: +0.070 R², +0.179 slope! Steps increase confirmed as bottleneck. Now approaching excellent threshold (0.95). 800 steps/frame is sufficient for 384-dim model.
Next: parent=6 (highest UCB=2.669), increase lr_NNR_f from 3E-5 to 4E-5 (prior optimal for Jp)

### Iter 7: excellent
Node: id=7, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=80000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25.0, batch_size=1, n_training_frames=100
Metrics: final_r2=0.960, final_mse=7.08E+00, total_params=445441, slope=0.848, training_time=5.0min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5 -> 4E-5
Visual: SSIM=1.000 perfect spatial match. Scatter shows tight cloud along diagonal with slope=0.848.
Parent rule: UCB selection - node 6 had highest UCB
Observation: EXCELLENT achieved! lr_NNR_f=4E-5 (prior knowledge optimal for Jp) improved R² 0.937→0.960 (+0.023) and slope 0.813→0.848 (+0.035). First excellent result in block.
Next: parent=7 (highest UCB=2.831), reduce omega_f from 25 to 20 to test if slope improves

