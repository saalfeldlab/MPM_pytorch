from typing import Optional, Literal, Annotated, Dict
import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Union

# Sub-config schemas for MPM_pytorch


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: int = 2
    n_frames: int = 1000
    start_frame: int = 0
    seed: int = 42

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Literal["periodic", "no", "periodic_special", "wall"] = "periodic"
    bounce: bool = False
    bounce_coeff: float = 0.1
    min_radius: float = 0.0
    max_radius: float = 0.1

    image_path: str = ""

    n_particles: int = 1000
    n_neurons: int = 1000
    n_input_neurons: int = 0
    n_excitatory_neurons: int = 0
    n_particles_max: int = 20000
    n_edges: int = 0
    max_edges: float = 1.0e6
    n_extra_null_edges: int = 0
    n_particle_types: int = 5
    n_neuron_types: int = 5
    baseline_value: float = -999.0
    n_particle_type_distribution: list[int] = [0]
    shuffle_particle_types: bool = False
    pos_init: str = "uniform"
    dpos_init: float = 0

    MPM_expansion_factor: float = 1.0
    MPM_n_objects: int = 9
    MPM_object_type: Literal['cubes', 'discs', 'spheres', 'stars', 'letters', 'gummy_bear'] = 'discs'
    MPM_gravity: float = -50
    MPM_rho_list: list[float] = [1.0, 1.0, 1.0]
    MPM_friction: float = 0.0
    MPM_young_coeff : float = 1.0
    MPM_surface_tension: float = 0.072
    MPM_tension_scaling: float = 1.0

    MPM_F_amplitude: float = 1.5
    MPM_C_amplitude: float = 200.0
    MPM_Jp_amplitude: float = 10.0

    diffusion_coefficients: list[list[float]] = None

    angular_sigma: float = 0
    angular_Bernouilli: list[float] = [-1]


    n_grid: int = 128

    n_nodes: Optional[int] = None
    n_node_types: Optional[int] = None
    node_coeff_map: Optional[str] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"
    node_proliferation_map: Optional[str] = None

    params: list[list[float]]
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    particle_model_name: str = ""
    cell_model_name: str = ""
    mesh_model_name: str = ""
    signal_model_name: str = ""
    prediction: Literal["first_derivative", "2nd_derivative"] = "2nd_derivative"
    integration: Literal["Euler", "Runge-Kutta"] = "Euler"

    field_type: str = ""
    field_grid: Optional[str] = ""

    input_size: int = 1
    output_size: int = 1
    hidden_dim: int = 1
    n_layers: int = 1

    input_size_2: int = 1
    output_size_2: int = 1
    hidden_dim_2: int = 1
    n_layers_2: int = 1

    input_size_decoder: int = 1
    output_size_decoder: int = 1
    hidden_dim_decoder: int = 1
    n_layers_decoder: int = 1


    lin_edge_positive: bool = False

    aggr_type: str

    mesh_aggr_type: str = "add"
    embedding_dim: int = 2
    embedding_init: str = ""

    update_type: Literal[
        "linear",
        "mlp",
        "pre_mlp",
        "2steps",
        "none",
        "no_pos",
        "generic",
        "excitation",
        "generic_excitation",
        "embedding_MLP",
        "test_field",
    ] = "none"

    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1
    init_update_gradient: bool = False

    kernel_type: str = "mlp"

    # INR type for external input learning
    # siren_t: input=t, output=n_particles (current implementation, works for n_particles < 100)
    # siren_id: input=(t, id), output=1 (scales better for large n_particles)
    # siren_txy: input=(t, x, y), output=1 (uses particle positions)
    # ngp: instantNGP hash encoding
    # lowrank: low-rank matrix factorization U @ V (not a neural network)
    inr_type: Literal["siren_t", "siren_id", "siren_txy", "ngp"] = "siren_t"

    input_size_nnr_f: int = 3
    n_layers_nnr_f: int = 5
    hidden_dim_nnr_f: int = 128
    output_size_nnr_f: int = 1
    outermost_linear_nnr_f: bool = True
    omega_f: float = 80.0
    omega_f_learning: bool = False  # make omega learnable during training
    use_layer_norm: bool = False  # add layer normalization to SIREN network

    nnr_f_xy_period: float = 1.0  # Spatial scaling (higher = expects slower spatial variation)
    nnr_f_T_period: float = 1.0  # Time scaling (higher = expects slower temporal variation)

    # InstantNGP (hash encoding) parameters
    ngp_n_levels: int = 24
    ngp_n_features_per_level: int = 2
    ngp_log2_hashmap_size: int = 22
    ngp_base_resolution: int = 16
    ngp_per_level_scale: float = 1.4
    ngp_n_neurons: int = 128
    ngp_n_hidden_layers: int = 4
    ngp_n_particles: int = 128  # Number of particles for NGP input (t, particle_id)


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str = "tab10"
    arrow_length: int = 10
    marker_size: int = 100
    xlim: list[float] = [-0.1, 0.1]
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]
    speedlim: list[float] = [0, 1]
    pic_folder: str = "none"
    pic_format: str = "jpg"
    pic_size: list[int] = [1000, 1100]
    data_embedding: int = 1


class ImageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_type: str = "none"
    cellpose_model: str = "cyto3"
    cellpose_denoise_model: str = ""
    cellpose_diameter: float = 30
    cellpose_flow_threshold: int = 0.4
    cellpose_cellprob_threshold: int = 0.0
    cellpose_channel: list[int] = [1]
    offset_channel: list[float] = [0.0, 0.0]
    tracking_file: str = ""
    trackmate_size_ratio: float = 1.0
    trackmate_frame_step: int = 1
    measure_diameter: float = 40.0


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"

    n_epochs: int = 20
    n_epochs_init: int = 99999
    epoch_reset: int = -1
    epoch_reset_freq: int = 99999
    batch_size: int = 1
    batch_ratio: float = 1
    small_init_batch_size: bool = True
    embedding_step: int = 1000
    shared_embedding: bool = False
    embedding_trial: bool = False
    remove_self: bool = True

    n_training_frames: int = 0

    pretrained_model: str = ""
    pre_trained_W: str = ""

    multi_connectivity: bool = False
    with_connectivity_mask: bool = False
    has_missing_activity: bool = False

    do_tracking: bool = False
    tracking_gt_file: str = ""
    ctrl_tracking: bool = False
    distance_threshold: float = 0.1
    epoch_distance_replace: int = 20

    denoiser: bool = False
    denoiser_type: Literal["none", "window", "LSTM", "Gaussian_filter", "wavelet"] = (
        "none"
    )
    denoiser_param: float = 1.0

    time_window: int = 0

    n_runs: int = 2
    seed: int = 42
    clamp: float = 0
    pred_limit: float = 1.0e10

    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: Literal["none", "tensor", "MLP"] = "none"
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Literal[
        "none",
        "replace_embedding",
        "replace_embedding_function",
        "replace_state",
        "replace_track",
    ] = "none"
    fix_cluster_embedding: bool = False
    cluster_method: Literal[
        "kmeans",
        "kmeans_auto_plot",
        "kmeans_auto_embedding",
        "distance_plot",
        "distance_embedding",
        "distance_both",
        "inconsistent_plot",
        "inconsistent_embedding",
        "none",
    ] = "distance_plot"
    cluster_distance_threshold: float = 0.01
    cluster_connectivity: Literal["single", "average"] = "single"

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_update_start: float = 0.0
    learning_rate_modulation_start: float = 0.0001
    learning_rate_W_start: float = 0.0001

    learning_rate_end: float = 0.0005
    learning_rate_embedding_end: float = 0.0001
    learning_rate_modulation_end: float = 0.0001
    Learning_rate_W_end: float = 0.0001

    learning_rate_missing_activity: float = 0.0001
    training_NNR_start_epoch: int = 0
    learning_rate_NNR_f: float = 0.0001
    learning_rate_omega_f: float = 0.0001
    coeff_omega_f_L2: float = 0.0
    training_NNR_start_epoch: int = 0
    total_steps: int = 50000  # INR training steps (Claude-tunable)
    n_iter_block: int = 16  # Claude block size (iterations per block)

    coeff_W_L1: float = 0.0
    coeff_W_L1_rate: float = 0.5
    coeff_W_L1_ghost: float = 0
    coeff_W_sign: float = 0

    coeff_entropy_loss: float = 0
    coeff_loss1: float = 1
    coeff_loss2: float = 1
    coeff_loss3: float = 1
    coeff_edge_diff: float = 10
    coeff_update_diff: float = 0
    coeff_update_msg_diff: float = 0
    coeff_update_msg_sign: float = 0
    coeff_update_u_diff: float = 0

    coeff_permutation: float = 100

    coeff_TV_norm: float = 0
    coeff_missing_activity: float = 0
    coeff_edge_norm: float = 0

    coeff_edge_weight_L1: float = 0
    coeff_edge_weight_L1_rate: float = 0.5
    coeff_phi_weight_L1: float = 0
    coeff_phi_weight_L1_rate: float = 0.5

    coeff_edge_weight_L2: float = 0
    coeff_phi_weight_L2: float = 0

    coeff_Jp_norm: float = 0
    coeff_F_norm: float = 0
    coeff_det_F: float = 1

    diff_update_regul: str = "none"

    coeff_model_a: float = 0
    coeff_model_b: float = 0
    coeff_lin_modulation: float = 0
    coeff_continuous: float = 0

    noise_level: float = 0
    measurement_noise_level: float = 0
    noise_model_level: float = 0
    loss_noise_level: float = 0.0


    rotation_augmentation: bool = False
    translation_augmentation: bool = False
    reflection_augmentation: bool = False
    velocity_augmentation: bool = False
    data_augmentation_loop: int = 40

    recursive_training: bool = False
    recursive_training_start_epoch: int = 0
    recursive_loop: int = 0
    coeff_loop: list[float] = [2, 4, 8, 16, 32, 64]
    time_step: int = 1
    recursive_sequence: str = ""
    recursive_parameters: list[float] = [0, 0]

    regul_matrix: bool = False
    sub_batches: int = 1
    sequence: list[str] = ["to track", "to cell"]

    MPM_trainer : str = "F"


# Claude exploration config
class ClaudeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: Literal["F", "Jp", "S", "C"] = "Jp"
    ucb_c: float = 1.414  # UCB exploration constant: UCB(k) = R²_k + c * sqrt(ln(N) / n_k)
    node_name: str = "a100"  # cluster GPU node: h100, a100, or l4


# Main config schema for MPM_pytorch


class MPM_pytorchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "MPM_pytorch"
    dataset: str
    data_folder_name: str = "none"
    connectome_folder_name: str = "none"
    data_folder_mesh_name: str = "none"
    config_file: str = "none"
    simulation: SimulationConfig
    graph_model: GraphModelConfig
    plotting: PlottingConfig
    training: TrainingConfig
    image_data: Optional[ImageData] = None
    claude: Optional[ClaudeConfig] = None

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return MPM_pytorchConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"  # Insert path to config file
    config = MPM_pytorchConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)