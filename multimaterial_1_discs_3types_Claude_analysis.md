# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1 (Iterations 1-12): siren_txy, Jp, n_frames=48

### Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: explore/initial
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.474, final_mse=107.13, total_params=12801, training_time=1.8min
Field: field_name=Jp, inr_type=siren_txy
Mutation: initial config
Parent rule: root (first iteration)
Observation: Poor fit - hidden_dim=64 too small for 9000 particles, omega_f=80 may cause instability
Next: parent=root, increase hidden_dim_nnr_f: 64 → 512

