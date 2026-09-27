# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matterix-compatible procedural rigid-object configuration."""

from dataclasses import field

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers.event_manager import EventTermCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG


marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
marker_cfg.prim_path = "/Visuals/ProceduralFrameTransformer"


@configclass
class ProceduralMatterixRigidObjectCfg(RigidObjectCfg):
    """RigidObjectCfg with the attributes Matterix expects.

    Unlike MatterixRigidObjectCfg, this class does not replace ``spawn`` with
    UsdFileCfg. Therefore, CuboidCfg, CylinderCfg and other procedural spawners
    can be used without external USD files.
    """

    pos: tuple[float, float, float] | None = None
    rot: tuple[float, float, float, float] | None = None

    activate_contact_sensors: bool = False

    # Use factories so instances do not share mutable dictionaries.
    event_terms: dict[str, EventTermCfg] = field(default_factory=dict)
    frames: dict[str, tuple[float, float, float] | OffsetCfg] = field(
        default_factory=dict
    )
    sensors: dict[str, FrameTransformerCfg] = field(default_factory=dict)

    semantic_tags: list[tuple[str, str]] = field(default_factory=list)
    semantics: list = field(default_factory=list)

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if self.pos is not None:
            self.init_state.pos = self.pos

        if self.rot is not None:
            self.init_state.rot = self.rot

        # Convert Matterix-style frame definitions into sensors.
        for frame_name, frame_value in self.frames.items():
            if isinstance(frame_value, OffsetCfg):
                offset = frame_value
            else:
                offset = OffsetCfg(pos=frame_value)

            self.sensors[frame_name] = FrameTransformerCfg(
                prim_path="",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="",
                        name=frame_name,
                        offset=offset,
                    )
                ],
            )