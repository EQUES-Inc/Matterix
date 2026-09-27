# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedurally generated bottle asset."""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ..procedural_rigid_object import ProceduralMatterixRigidObjectCfg


@configclass
class BOTTLE_INST_CFG(ProceduralMatterixRigidObjectCfg):
    """Simple cylindrical bottle that requires no external USD."""

    prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"

    spawn = sim_utils.CylinderCfg(
        radius=0.032,
        height=0.14,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.20,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.55, 0.85),
            roughness=0.35,
        ),
        semantic_tags=[
            ("class", "bottle"),
        ],
    )

    frames = {
        "pre_grasp": (0.0, 0.0, 0.10),
        "grasp": (0.0, 0.0, 0.02),
        "post_grasp": (0.0, 0.0, 0.14),
    }

    semantic_tags = [("class", "bottle")]


BOTTLE_CFG = BOTTLE_INST_CFG