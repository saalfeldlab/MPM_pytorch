# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|

### Established Principles

### Open Questions
- Q1: Does this dataset (multimaterial_1_discs_3types) have similar optimal configs to prior experiments?
- Q2: What is the capacity requirement for Jp@100frames on this dataset?

---

## Previous Block Summary

---

## Current Block (Block 1)

### Block Info
Field: Jp, inr_type=siren_txy, n_frames=100
Iterations: 1-12

### Hypothesis
Apply prior knowledge: Jp optimal at omega_f=20-25, hidden_dim=384, n_layers=3 for 100 frames.
Starting config has omega_f=80 and hidden_dim=64, both far off. First fix omega_f.

### Iterations This Block

## Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80, batch_size=1
Metrics: final_r2=0.420, final_mse=1.22E+02, total_params=12801, training_time=2.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: [initial config]
Observation: omega_f=80 way too high, hidden_dim=64 way too small. slope=0.049 = severe underprediction.
Next: parent=1, omega_f: 80 -> 25

## Iter 2: poor
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.427, final_mse=1.24E+02, total_params=12801, training_time=2.6min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 80 -> 25
Observation: omega_f fix had minimal effect (R² +0.007). Capacity is the clear bottleneck - hidden_dim=64 is 6x too small.
Next: parent=2, hidden_dim_nnr_f: 64 -> 384

## Iter 3: moderate
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.830, final_mse=7.62E+01, total_params=445441, training_time=3.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 64 -> 384
Visual: GT/Pred spatial patterns match but amplitude collapsed (slope=0.254)
Observation: Capacity increase gave +0.40 R². But slope=0.254 = severe underprediction. lr=1E-5 too conservative.
Next: parent=3, lr_NNR_f: 1E-5 -> 3E-5

## Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.862, final_mse=2.83E+01, total_params=445441, training_time=3.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5 -> 3E-5
Observation: lr increase improved R² (+0.032) and slope doubled (0.254→0.611). Slope still <1, need higher lr.
Next: parent=4, lr_NNR_f: 3E-5 -> 5E-5

## Iter 5: good
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.927, final_mse=1.32E+01, total_params=445441, training_time=3.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5 -> 5E-5
Visual: GT/Pred match well, slope=0.773 still underpredicting but improved from 0.611
Observation: lr 5E-5 gave +0.065 R², +0.16 slope. Clear monotonic trend: higher lr → better. Target R²>0.95 not yet reached.
Next: parent=5, lr_NNR_f: 5E-5 -> 8E-5

### Emerging Observations
- Initial config severely misconfigured: omega_f 3-4x too high, hidden_dim 6x too small
- omega_f fix (80→25) had minimal impact when capacity-limited (R² +0.007)
- hidden_dim fix (64→384) gave major boost (+0.40 R²) - capacity was primary bottleneck
- lr scaling map: 1E-5 (0.830, slope=0.254) → 3E-5 (0.862, slope=0.611) → 5E-5 (0.927, slope=0.773)
- Clear monotonic lr trend: every lr increase improves both R² and slope
- Next probe: lr=8E-5 (expected to approach R²>0.95 target based on trend)

