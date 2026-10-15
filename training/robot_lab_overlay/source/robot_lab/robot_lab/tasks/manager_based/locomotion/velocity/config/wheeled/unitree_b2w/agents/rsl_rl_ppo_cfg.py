# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UnitreeB2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2w_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2WFlatPPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_b2w_flat"


@configclass
class UnitreeB2WZ1EthFinetunePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Fine-tuning ETH's SRU policy (eth_finetune_env_cfg.py): its exact network -- LSTM 256 into
    256-256-128-64 -- started from its weights by training/train_eth_ft.py, so a small learning
    rate, little exploration noise and entropy: change the stance, keep the driving."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "b2w_z1_eth_finetune"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.15,
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128, 64],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
