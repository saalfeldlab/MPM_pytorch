# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1 - siren_txy, Field Jp, 48 frames

### Iter 1: [poor]
Node: id=1, parent=root
Mode/Strategy: exploit (initial)
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.471, final_mse=107.1, total_params=12801, compression_ratio=70.3, training_time=2.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: initialization from root
Parent rule: UCB file shows only node 1; starting fresh block
Observation: Catastrophic failure - model far too small (64 hidden) and omega_f far too high (80.0 vs typical 20-50). Both factors likely contribute to poor fit.
Next: parent=1

