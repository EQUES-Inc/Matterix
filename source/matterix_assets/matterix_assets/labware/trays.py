# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedurally generated tray asset."""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ..procedural_rigid_object import ProceduralMatterixRigidObjectCfg


@configclass
class TRAY_INST_CFG(ProceduralMatterixRigidObjectCfg):
    """Simple solid tray approximation requiring no external USD."""

    prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"

    spawn = sim_utils.CuboidCfg(
        size=(0.24, 0.16, 0.025),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.50,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.35, 0.38),
            metallic=0.25,
            roughness=0.45,
        ),
        semantic_tags=[
            ("class", "tray"),
        ],
    )

    frames = {
        "pre_grasp": (0.0, 0.0, 0.08),
        "grasp": (0.0, 0.0, 0.015),
        "post_grasp": (0.0, 0.0, 0.12),
        "pre_place": (0.0, 0.0, 0.10),
        "place": (0.0, 0.0, 0.02),
        "post_place": (0.0, 0.0, 0.12),
    }

    semantic_tags = [("class", "tray")]


TRAY_CFG = TRAY_INST_CFG