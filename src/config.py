import dataclasses
from rltools.config import Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class MPOConfig(Config):
    """
    Args: #noqa
        discount: MDP discount factor.
        action_repeat: repeat an action for multiple timesteps.
        n_step: multistep update w/o off-policy corrections.
        num_actions: number of actions in the MPO estimation step.
        num_actor_quantiles: number of quantiles in the distributional policy improvement.
        num_critic_quantiles: number of quantiles in the distributional policy learning.
        hubber_delta: threshold for switching between L1 / L2 loss.
        tv_constraint: clipping in the MPO E-step to prevent rapid policy changes.
        epsilon_eta:  target KL divergence in the MPO E-step.
        epsilon_mean: target KL in MPO M-step for mean distribution.
        epsilon_std: target KL in MPO M-step for stddev distribution.
        init_log_temperatrue: initial value of MPO E-step loss dual parameter.
        init_log_alpha_mean: initial value of MPO M-step mean KL dual parameter.
        init_log_alpha_std: initial value of MPO M-step std KL dual parameter.
        hindsight_goal_key: source of hindsight goal relabelling.
        augmentation_strategy: (none, final, future) as in the HER paper.
        num_augmentations: number of trajectory augmentations.
        activation: which activation to use for all the networks.
        normalization: preactivation normalization to use everywhere.
        keys: valid observations regex filter.
        mlp_layers: encoder's mlp layers.
        pn_number: number of points per point cloud observation.
        img_size: pixels observations shape.
        pn_layers: encoder's point net layers.
        cnn_depths: encoder's cnn depths.
        cnn_kernels: encoder's cnn kernels.
        cnn_strides: encoder's cnn strides.
        feature_fusion: regex filter for observations which should be concatenated with lowdim features.
        actor_backend: gpu or cpu jax.backend.
        actor_layers: actor's MLP hidden layers.
        min_std: minimal stddev value of actor's normal policy.
        init_std: initial stddev value of actor's normal policy.
        use_iqn: ordinary or distributional critic choice.
        num_critic_heads: number of critics heads.
        critic_layers: critic's MLP hidden layers.
        quantile_embedding_dim: same as in the IQN paper.
        min_replay_size: required replay size for sampling to begin.
        samples_per_insert: number of gradient steps per one environment step.
        batch_size: learner batch_size.
        buffer_capacity: maximum replay buffer size.
        actor_update_every: number of steps before fetching new weights.
        learner_dump_every: number of gradient steps before saving a replay buffer.
        reverb_port: reverb.Server port.
        learning_rate: common for all the networks.
        dual_lr: dual params learning rate.
        adam_b1: optax.adamw b1.
        adam_b2: optax.adamw b2.
        adam_eps: optax.adamw epsion.
        weight_decay: optax.adamw weight decay.
        target_actor_update_period: target actor hard update period.
        target_critic_update_period: target critic hard update period.
        max_seq_len: max env interaction sequence length before sending transitions to a replay buffer.
        eval_every: actor evaluation interval.
        log_every: learner logging interval.
        eval_times: number of episodes on evalution.
        grad_norm: global norm grad clipping.
        mp_policy: mixed precision policy.
        jit: to jit or not learner step.
        num_actors: number of distributed actors.
        seed: random seed.
        task: rl task.
        logdir: logging dir.
        total_steps: maximum number of interactions with the env.
        time_limit: impose env maximum episode length.
    """

    # Algorithm
    discount: float = .99
    action_repeat: int = 1
    n_step: int = 5
    #  IQN.
    num_actions: int = 20
    num_actor_quantiles: int = 32
    num_critic_quantiles: int = 4
    hubber_delta: float = 1.
    #  MPO.
    tv_constraint: float = 1.
    epsilon_eta: float = .1
    epsilon_mean: float = 2.5e-3
    epsilon_std: float = 1e-6
    init_log_temperature: float = 10.
    init_log_alpha_mean: float = 10.
    init_log_alpha_std: float = 1000.
    #  HER.
    goal_sources: tuple[str, ...] = ("kinect", "box/position")
    goal_targets: tuple[str, ...] = ("goal_image", "goal_pos")
    augmentation_strategy: str = "final"
    num_augmentations: int = 1

    # Architecture
    activation: str = "elu"
    normalization: str = "layer"
    #   Encoder
    keys: str = r"(ur5e|goal_pos|box).*"
    mlp_layers: Layers = ()
    pn_number: int = 1000
    img_size: tuple[int, int] = (84, 84)
    pn_layers: Layers = (64, 128, 256)
    cnn_depths: Layers = (32, 64, 64)
    cnn_kernels: Layers = (8, 4, 3)
    cnn_strides: Layers = (4, 2, 1)
    feature_fusion: str = r"$^"
    #   Actor
    actor_backend: str = "cpu"
    actor_layers: Layers = (256, 256, 256)
    min_std: float = .1
    init_std: float = .5
    #   Critic
    use_iqn: bool = False
    num_critic_heads: int = 2
    critic_layers: Layers = (512, 512, 256)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 5e3
    samples_per_insert: int = 32
    batch_size: int = 256
    buffer_capacity: int = 1e6
    actor_update_every: int = 1
    learner_dump_every: int = 40_000
    reverb_port: int = 4445

    # training
    learning_rate: float = 1e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-6
    weight_decay: float = 1e-6
    target_actor_update_period: int = 100
    target_critic_update_period: int = 100
    max_seq_len: int = 40
    eval_every: int = 1e4
    log_every: int = 1e2
    eval_times: int = 15
    grad_norm: float = 40.
    mp_policy: str = "p=f32,c=f32,o=f32"
    jit: bool = True
    num_actors: int = 5

    # task
    seed: int = 0
    task: str = "src_lift"
    logdir: str = "logdir/src_lift_features_no_dr"
    total_steps: int = 1e9
    time_limit: int = 200
    discretize: bool = True
    nbins: int = 11
    use_ordinal: bool = False
