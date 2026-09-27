# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedurally generated flask asset."""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ..procedural_rigid_object import ProceduralMatterixRigidObjectCfg


@configclass
class FLASK_INST_CFG(ProceduralMatterixRigidObjectCfg):
    """Simple conical flask approximation requiring no external USD."""

    prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"

    spawn = sim_utils.ConeCfg(
        radius=0.055,
        height=0.13,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.25,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            # diffuse_color=(0.45, 0.75, 0.75),
            diffuse_color=(1.0, 0.0, 0.0),
            roughness=0.30,
        ),
        semantic_tags=[
            ("class", "flask"),
        ],
    )

    frames = {
        "pre_grasp": (0.0, 0.0, 0.10),
        "grasp": (0.0, 0.0, 0.025),
        "post_grasp": (0.0, 0.0, 0.14),
    }

    semantic_tags = [("class", "flask")]


FLASK_CFG = FLASK_INST_CFG