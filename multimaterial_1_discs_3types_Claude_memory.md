# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | Regime | Best R² | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Key finding |
|-------|--------|---------|------------------|--------------------|--------------------|-----------------|-------------|

### Established Principles

### Open Questions
- omega_f=80 + hidden_dim=64: possible cause of poor R²?
- siren_txy on Jp field: what is optimal configuration?

---

## Previous Block Summary

---

## Current Block (Block 1)

### Block Info
Field: Jp, INR type: siren_txy
Iterations: 1-12

### Hypothesis
Starting from small network (hidden_dim=64) with high omega_f=80. Expect increasing hidden_dim will improve R² significantly.

### Iterations This Block

## Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: exploit (first iteration)
Config: lr_NNR_f=1E-5, total_steps=100000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.734, final_mse=9.04e+01, total_params=12801
Field: field_name=Jp, inr_type=siren_txy
Mutation: initial config (root)
Parent rule: N/A (first iteration)
Observation: Very small hidden_dim (64) likely underfitting. High omega_f (80) may cause instability.
Next: parent=1

## Iter 2: good
Node: id=2, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=100000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.922, final_mse=3.62e+01, total_params=198657
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 64→256
Parent rule: Parent=1 (highest UCB), exploit strategy
Observation: 4x increase in hidden_dim yielded +0.188 R² improvement. Network capacity was the bottleneck.
Next: parent=2

### Emerging Observations
- hidden_dim scaling: 64→256 improved R² from 0.734 to 0.922 (+0.188)
- omega_f=80 is tolerable with larger networks (did not cause instability with hidden_dim=256)
- Network capacity appears to be the dominant factor for siren_txy on Jp field
- Next step: test hidden_dim=512 to target R²>0.95

