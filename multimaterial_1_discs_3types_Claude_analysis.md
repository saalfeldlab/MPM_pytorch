# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1: Jp field, siren_txy, 100 frames

### Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: exploit (first iteration)
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80, batch_size=1
Metrics: final_r2=0.399, final_mse=121.7, total_params=12801, training_time=6.3min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: Initial config (from template)
Parent rule: root (first iteration)
Observation: Severe underfitting - hidden_dim=64 is grossly under-capacity for 9000 particles. omega_f=80 is too high for Jp. Model predicts near-constant ~1.0.
Visual: Complete mismatch - prediction shows near-constant field while GT has rich structure. Slope=0.049 indicates severe underprediction. Scatter cloud vertical (no correlation).
Next: parent=1, increase hidden_dim dramatically (64→384) and reduce omega_f (80→25)

### Iter 2: moderate
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.835, final_mse=75.6, slope=0.257, total_params=445441, training_time=5.9min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: hidden_dim 64→384, omega_f 80→25
Parent rule: UCB selection (Node 2=1.835 highest)
Observation: Capacity increase dramatically improved R² (0.399→0.835), but slope=0.257 indicates severe underprediction. Model has capacity but LR=1E-5 is too conservative for proper convergence.
Next: parent=2, increase lr 1E-5→4E-5 (prior knowledge: Jp optimal ~4E-5, multimaterial tolerates higher LR)

### Iter 3: moderate
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.913, final_mse=19.4, slope=0.683, total_params=445441, training_time=5.3min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: lr 1E-5→4E-5
Parent rule: UCB selection (Node 2 highest UCB=1.835)
Observation: LR increase improved R² (0.835→0.913, +0.078) and slope dramatically (0.257→0.683, +0.426). Model now predicting actual values, not near-constant. Prior knowledge about higher LR for multimaterial confirmed.
Next: parent=3, probe LR upper boundary: lr 4E-5→6E-5 (prior: multimaterial tolerates 2x higher LR, optimal may be ~8E-5)

### Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=6E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.942, final_mse=9.67, slope=0.825, total_params=445441, training_time=8.6min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: lr 4E-5→6E-5
Parent rule: UCB selection (Node 4 highest UCB=2.356)
Observation: LR increase improved R² (0.913→0.942, +0.029) and slope (0.683→0.825, +0.142). Diminishing returns on R² improvement but slope still improving. Per-frame MSE shows higher error at early frames (0-40).
Visual: GT/Pred spatial patterns similar but Pred shows lower contrast and some artifacts. Scatter plot confirms slope<1 underprediction especially at high GT values (>1.5).
Next: parent=4, continue LR probe: lr 6E-5→8E-5 (still room for improvement based on slope<1)

### Iter 5: moderate
Node: id=5, parent=4
Mode/Strategy: boundary-probe (LR upper boundary)
Config: lr_NNR_f=8E-5, total_steps=50000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.892, final_mse=14.59, slope=0.863, total_params=445441, training_time=8.6min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: lr 6E-5→8E-5
Parent rule: UCB selection (Node 4 highest UCB=1.996)
Observation: **LR BOUNDARY FOUND.** R² REGRESSED (0.942→0.892, -0.05) while slope marginally improved (0.825→0.863, +0.038). lr=8E-5 overshoots optimal. LR map: 1E-5(0.835) < 4E-5(0.913) < 6E-5(0.942) > 8E-5(0.892). Optimal lr=6E-5 confirmed.
Visual: Not available
Next: parent=4, exploit best config with capacity increase: hidden_dim 384→512

### Iter 6: moderate
Node: id=6, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=6E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.918, final_mse=12.38, slope=0.822, total_params=790529, training_time=7.7min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: hidden_dim 384→512
Parent rule: UCB selection (Node 4 highest UCB)
Observation: **CAPACITY INCREASE REGRESSED R²** (0.942→0.918, -0.024). Slope unchanged (0.825→0.822). hidden_dim=384 appears optimal; 512 may overfit or require more steps. Prior knowledge (Jp ceiling at 384) confirmed.
Visual: GT/Pred spatial patterns reasonably similar but some artifacts. Per-frame MSE higher at early frames (0-40), lower at middle frames. Scatter shows underprediction (slope=0.82<1).
Next: parent=6, try total_steps 50k→100k (500 steps/frame → 1000 steps/frame per prior knowledge requirement)

### Iter 7: moderate
Node: id=7, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=6E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.961, final_mse=5.86, slope=0.890, total_params=790529, training_time=10.4min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: total_steps 50k→100k
Parent rule: UCB selection (Node 6 highest after parent=4 with steps experiment)
Observation: **STEPS INCREASE HIGHLY EFFECTIVE.** R² improved significantly (0.918→0.961, +0.043) and slope improved (0.822→0.890, +0.068). hidden_dim=512 NOT a problem when given enough training steps. 1000 steps/frame is the key requirement.
Visual: Not available (file not found)
Next: parent=7, increase total_steps 100k→150k (test if 1500 steps/frame can push R²>0.99)

### Iter 8: good
Node: id=8, parent=7
Mode/Strategy: exploit
Config: lr_NNR_f=6E-5, total_steps=150000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=25, batch_size=1
Metrics: final_r2=0.979, final_mse=3.10, slope=0.930, total_params=790529, training_time=16.8min
Field: field_name=Jp, inr_type=siren_txy, n_frames=100
Mutation: total_steps 100k→150k
Parent rule: UCB selection (Node 7 highest after steps experiment)
Observation: **MORE STEPS CONTINUES TO IMPROVE.** R² +0.018 (0.961→0.979), slope +0.040 (0.890→0.930). 1500 steps/frame effective. Loss curve still improving at 150k steps - not yet plateaued. Steps/frame scaling: 500→0.918, 1000→0.961, 1500→0.979.
Visual: GT/Pred spatial patterns match well. Scatter tight along diagonal with some outliers at high GT values (slope=0.93 indicates slight underprediction). Per-frame MSE high at early frames but converges to near-zero by frame 50.
Next: parent=8, test omega_f 25→20 (lower frequency may help remaining error)

