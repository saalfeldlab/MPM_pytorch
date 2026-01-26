# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1: Jp@100frames, siren_txy
Hypothesis: Establish baseline for this dataset using prior knowledge optimal configs

## Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80, batch_size=1
Metrics: final_r2=0.420, final_mse=1.22E+02, total_params=12801, training_time=2.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: [initial config] omega_f=80 and hidden_dim=64 both far from prior optimal
Parent rule: initial exploration
Observation: omega_f=80 is 3-4x too high for 100 frames (prior: 20-25). hidden_dim=64 is 6x too small (prior: 384). Slope=0.049 indicates severe underprediction.
Next: parent=1, mutate omega_f: 80 -> 25 (most critical fix)

## Iter 2: poor
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.427, final_mse=1.24E+02, total_params=12801, training_time=2.6min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 80 -> 25
Parent rule: fix most critical misconfiguration (omega_f) first
Observation: omega_f fix had minimal effect (R² 0.420→0.427). Slope=0.040 still catastrophic. Capacity is the bottleneck - hidden_dim=64 is far too small (prior: 384).
Next: parent=2, mutate hidden_dim_nnr_f: 64 -> 384 (address capacity bottleneck)

## Iter 3: moderate
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.830, final_mse=7.62E+01, total_params=445441, training_time=3.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 64 -> 384
Parent rule: address capacity bottleneck (hidden_dim 6x too small per prior knowledge)
Visual: GT/Pred spatial patterns match (disc structure visible), but scatter shows severe amplitude collapse - predictions cluster around 1.0 while GT spans 0-2.0
Observation: Capacity increase boosted R² dramatically (+0.40). BUT slope=0.254 reveals model not learning dynamic range. lr=1E-5 may be too conservative - prior says this dataset tolerates 2x higher lr.
Next: parent=3, mutate lr_NNR_f: 1E-5 -> 3E-5 (increase learning rate per dataset-specific prior)

## Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.862, final_mse=2.83E+01, total_params=445441, training_time=3.4min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5 -> 3E-5
Parent rule: increase lr per dataset-specific prior (this dataset tolerates 2x higher lr)
Observation: lr increase improved R² (+0.032) and slope doubled (0.254→0.611). MSE dropped 2.7x. Slope still <1, indicating model can benefit from higher lr. Prior suggests optimal lr may be 5E-5 to 8E-5 for this dataset.
Next: parent=4, mutate lr_NNR_f: 3E-5 -> 5E-5 (continue probing lr upper boundary)

## Iter 5: good
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.927, final_mse=1.32E+01, total_params=445441, training_time=3.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 3E-5 -> 5E-5
Parent rule: continue lr optimization (prior suggests dataset-specific optimal at 5E-5 to 8E-5)
Visual: GT/Pred spatial patterns match well, SSIM values not shown but spatial structure preserved. Scatter shows slope=0.773, still underpredicting high values. Loss curve shows good convergence.
Observation: lr increase 3E-5→5E-5 improved R² (+0.065) and slope (+0.16). Clear monotonic trend: higher lr → better fit. Still below R²>0.95 target. Continue probing lr upper boundary.
Next: parent=5, mutate lr_NNR_f: 5E-5 -> 8E-5 (probe upper boundary per prior knowledge)

