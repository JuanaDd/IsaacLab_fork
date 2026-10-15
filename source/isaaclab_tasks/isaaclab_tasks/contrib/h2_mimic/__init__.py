# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree H2 closed-loop motion imitation with Newton VBD."""

import gymnasium as gym

for suffix, config in (("", "H2MimicEnvCfg"), ("-Eval", "H2MimicPlayEnvCfg")):
    gym.register(
        id=f"IsaacContrib-H2-Closeloop-Mimic-VBD{suffix}",
        entry_point=f"{__name__}.env:H2MimicEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.configs:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.configs:H2MimicPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
