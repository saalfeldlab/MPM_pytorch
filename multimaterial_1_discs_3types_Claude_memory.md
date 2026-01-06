# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|

### Established Principles

### Open Questions

---

## Previous Block Summary

---

## Current Block (Block 1)

### Block Info
Field: field_name=Jp, inr_type=siren_txy, n_frames=48
Iterations: 1 to 12

### Hypothesis
Baseline siren_txy architecture needs proper capacity (hidden_dim ≥ 256) and moderate omega_f (20-50) to achieve good R² on Jp field.

### Iterations This Block

#### Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: explore/initial
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.474, final_mse=107.13, total_params=12801, training_time=1.8min
Mutation: initial config
Observation: Poor fit - hidden_dim=64 too small for 9000 particles, omega_f=80 may cause instability
Next: parent=root, increase hidden_dim_nnr_f: 64 → 512

### Emerging Observations
- Initial config with hidden_dim=64 and omega_f=80 gives poor R²=0.474
- Need to test: (1) larger hidden_dim, (2) lower omega_f
- Priority: increase capacity first via hidden_dim

