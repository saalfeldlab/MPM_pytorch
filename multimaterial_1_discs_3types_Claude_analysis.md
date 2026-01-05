# Experiment Log: multimaterial_1_discs_3types_Claude

---

## Block 1: siren_txy on Jp field

### Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: exploit (first iteration)
Config: lr_NNR_f=1E-5, total_steps=100000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.734, final_mse=9.04e+01, total_params=12801
Field: field_name=Jp, inr_type=siren_txy
Mutation: initial config (root)
Parent rule: N/A (first iteration)
Observation: Very small hidden_dim (64) likely underfitting. High omega_f (80) may cause instability.
Next: parent=1, mutation=hidden_dim 64→256

### Iter 2: good
Node: id=2, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.922, final_mse=3.62e+01, total_params=198657
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 64→256
Parent rule: Parent=1 (highest UCB), exploit strategy
Observation: 4x increase in hidden_dim yielded +0.188 R² improvement. Network capacity was the bottleneck. omega_f=80 still tolerable.
Next: parent=2, mutation=hidden_dim 256→512 (target R²>0.95)

