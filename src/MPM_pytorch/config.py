from typing import Optional, Literal, Annotated, Dict
import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Union

# Sub-config schemas for ParticleGraph


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: int = 2
    n_frames: int = 1000
    start_frame: int = 0

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Literal["periodic", "no", "periodic_special", "wall"] = "periodic"
    bounce: bool = False
    bounce_coeff: float = 0.1
    min_radius: float = 0.0
    max_radius: float = 0.1

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
    MPM_object_type: Literal['cubes', 'discs', 'spheres', 'stars', 'letters'] = 'discs'
    MPM_gravity: float = -50
    MPM_rho_list: list[float] = [1.0, 1.0, 1.0]
    MPM_friction: float = 0.0
    MPM_young_coeff : float = 1.0

    diffusion_coefficients: list[list[float]] = None

    angular_sigma: float = 0
    angular_Bernouilli: list[float] = [-1]

    n_grid: int = 128

    n_nodes: Optional[int] = None
    n_node_types: Optional[int] = None
    node_coeff_map: Optional[str] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"
    node_proliferation_map: Optional[str] = None

    adjacency_matrix: str = ""

    short_term_plasticity_mode: str = "depression"

    connectivity_file: str = ""
    connectivity_init: list[float] = [-1]
    connectivity_filling_factor: float = 1
    connectivity_type: Literal["none", "distance", "voronoi", "k_nearest"] = "distance"
    connectivity_parameter: float = 1.0
    connectivity_distribution: str = "Gaussian"
    connectivity_distribution_params: float = 1

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    params: list[list[float]]
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    cell_cycle_length: list[float] = [-1]
    cell_death_rate: list[float] = [-1]
    cell_area: list[float] = [-1]
    cell_type_map: Optional[str] = None
    final_cell_mass: list[float] = [-1]
    pos_rate: list[list[float]] = None
    neg_rate: list[list[float]] = None
    has_cell_division: bool = False
    has_cell_death: bool = False
    has_cell_state: bool = False
    non_discrete_level: float = 0
    cell_active_model_coeff: float = 1
    cell_inert_model_coeff: float = 0
    coeff_area: float = 1
    coeff_perimeter: float = 0
    kill_cell_leaving: bool = False

    state_type: Literal["discrete", "sequence", "continuous"] = "discrete"
    state_params: list[float] = [-1]


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

    multi_mlp_params: List[List[Union[int, int, int, int, str]]] = None

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

    kernel_type: str = "mlp"

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1
    outermost_linear_nnr: bool = True
    omega: float = 80.0

    input_size_modulation: int = 2
    n_layers_modulation: int = 3
    hidden_dim_modulation: int = 64
    output_size_modulation: int = 1

    input_size_excitation: int = 3
    n_layers_excitation: int = 5
    hidden_dim_excitation: int = 128

    excitation_dim: int = 1


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
    seed: int = 40
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

    learning_rate_NNR: float = 0.0001
    learning_rate_missing_activity: float = 0.0001

    coeff_L1: float = 0.0
    coeff_anneal_L1: float = 0
    coeff_entropy_loss: float = 0
    coeff_loss1: float = 1
    coeff_loss2: float = 1
    coeff_loss3: float = 1
    coeff_edge_diff: float = 10
    coeff_update_diff: float = 0
    coeff_update_msg_diff: float = 0
    coeff_update_msg_sign: float = 0
    coeff_update_u_diff: float = 0
    coeff_sign: float = 0
    coeff_permutation: float = 100
    coeff_L1_ghost: float = 0
    coeff_sign: float = 0
    coeff_TV_norm: float = 0
    coeff_missing_activity: float = 0
    coeff_edge_norm: float = 0
    coeff_edge_weight_L1: float = 0

    diff_update_regul: str = "none"

    coeff_model_a: float = 0
    coeff_model_b: float = 0
    coeff_lin_modulation: float = 0
    coeff_continuous: float = 0

    noise_level: float = 0
    measurement_noise_level: float = 0
    noise_model_level: float = 0
    rotation_augmentation: bool = False
    translation_augmentation: bool = False
    reflection_augmentation: bool = False
    velocity_augmentation: bool = False
    data_augmentation_loop: int = 40

    recursive_loop: int = 0
    time_step: int = 1
    recursive_sequence: str = ""
    recursive_parameters: list[float] = [0, 0]

    regul_matrix: bool = False
    sub_batches: int = 1
    sequence: list[str] = ["to track", "to cell"]

    MPM_trainer : str = "F"


# Main config schema for ParticleGraph


class ParticleGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "ParticleGraph"
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

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return ParticleGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"  # Insert path to config file
    config = ParticleGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)
