# Local addition (not upstream robot_lab): Z1 arm reaching on the B2W, standing.

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Reach-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:B2WZ1ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1ReachPPORunnerCfg",
    },
)
