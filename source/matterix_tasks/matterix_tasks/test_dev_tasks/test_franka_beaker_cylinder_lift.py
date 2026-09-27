# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test development environment with Franka robot and switchable gripper."""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg

from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.labware.bottles import BOTTLE_INST_CFG
from matterix_assets.labware.flasks import FLASK_INST_CFG
from matterix_assets.labware.trays import TRAY_INST_CFG


from matterix_assets.robots import (
    FRANKA_PANDA_HIGH_PD_IK_CFG,
    FRANKA_LLINK_ROBOTIQ2F85_CFG
)

# Workflow definitions
from matterix_sm import PickObjectCfg
from matterix_sm.robot_action_spaces import (
    FRANKA_IK_ACTION_SPACE,
)

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


# =============================================================================
# Gripper selection
# =============================================================================

# True  -> Franka Panda + Robotiq 2F-85
# False -> Franka Panda + standard Panda gripper
USE_ROBOTIQ85 = False


if USE_ROBOTIQ85:
    ROBOT_CFG = FRANKA_LLINK_ROBOTIQ2F85_CFG

    GRIPPER_JOINT_NAMES = [
        "robotiq_85_left_knuckle_joint",
        "robotiq_85_right_knuckle_joint",
    ]

    ACTION_SPACE_INFO = FRANKA_IK_ACTION_SPACE

    WRIST_CAMERA_PARENT = "virtual_eef_link"
    WRIST_CAMERA_POS = (0.00, 0.00, 0.05)
    WRIST_CAMERA_ROT = (0.7071, 0.0, -0.7071, 0.0)

else:
    ROBOT_CFG = FRANKA_PANDA_HIGH_PD_IK_CFG

    GRIPPER_JOINT_NAMES = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    ACTION_SPACE_INFO = FRANKA_IK_ACTION_SPACE

    WRIST_CAMERA_PARENT = "panda_hand"
    WRIST_CAMERA_POS = (0.00, 0.00, 0.05)
    WRIST_CAMERA_ROT = (1.0, 0.0, 0.0, 0.0)


# =============================================================================
# Event configs
# =============================================================================

@configclass
class EventCfg(EventManagerCfg):
    """Configuration for randomization events."""

    # Reset scene to default state
    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )

    # Controlled position jitter for the two target objects.
    #
    # Important:
    # - Keep the jitter small so grasp difficulty stays comparable across episodes.
    # - The collector should reuse the same reset seed for the beaker/cylinder pair.
    # - To remove the "left means beaker" shortcut completely, use a custom reset
    #   event that swaps the two anchors 50/50. With only reset_root_state_uniform,
    #   the two objects keep their own anchor regions.
    randomize_beaker_position = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("beaker"),
        },
    )

    randomize_blue_cylinder_position = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("blue_cylinder"),
        },
    )


# =============================================================================
# Observation configs
# =============================================================================

@configclass
class ObservationManagerCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ArticulationsGroup(ObsGroup):
        """Robot observations."""

        robot__root_world_pos = ObsTerm(
            func=mdp.root_world_pos,
            params={"asset_name": "robot"},
        )

        robot__root_world_quat = ObsTerm(
            func=mdp.root_world_quat,
            params={"asset_name": "robot"},
        )

        robot__joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_name": "robot"},
        )

        robot__joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_name": "robot"},
        )

        robot__ee_world_pos = ObsTerm(
            func=mdp.ee_world_pos,
            params={"asset_name": "robot"},
        )

        robot__ee_world_quat = ObsTerm(
            func=mdp.ee_world_quat,
            params={"asset_name": "robot"},
        )

        robot__gripper_pos = ObsTerm(
            func=mdp.gripper_pos,
            params={"asset_name": "robot"},
        )

        robot__grasping_frame_world_pos = ObsTerm(
            func=mdp.frame_world_pos,
            params={
                "asset_name": "robot",
                "frame_name": "grasping_frame",
            },
        )

        robot__grasping_frame_world_quat = ObsTerm(
            func=mdp.frame_world_quat,
            params={
                "asset_name": "robot",
                "frame_name": "grasping_frame",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        """Rigid object observations."""

        beaker__object_world_pos = ObsTerm(
            func=mdp.object_world_pos,
            params={"asset_name": "beaker"},
        )

        beaker__object_world_quat = ObsTerm(
            func=mdp.object_world_quat,
            params={"asset_name": "beaker"},
        )

        beaker__object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel,
            params={"asset_name": "beaker"},
        )

        beaker__object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel,
            params={"asset_name": "beaker"},
        )

        beaker__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={
                "asset_name": "beaker",
                "frame_name": "pre_grasp",
            },
        )

        beaker__grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={
                "asset_name": "beaker",
                "frame_name": "grasp",
            },
        )

        beaker__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={
                "asset_name": "beaker",
                "frame_name": "post_grasp",
            },
        )

        blue_cylinder__object_world_pos = ObsTerm(
            func=mdp.object_world_pos,
            params={"asset_name": "blue_cylinder"},
        )

        blue_cylinder__object_world_quat = ObsTerm(
            func=mdp.object_world_quat,
            params={"asset_name": "blue_cylinder"},
        )

        blue_cylinder__object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel,
            params={"asset_name": "blue_cylinder"},
        )

        blue_cylinder__object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel,
            params={"asset_name": "blue_cylinder"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


# =============================================================================
# Environment config
# =============================================================================

@configclass
class FrankaBeakerLiftEnvTestCfg(MatterixBaseEnvCfg):
    """Franka lifting environment with switchable Panda / Robotiq gripper."""

    env_spacing = 10.0

    # -------------------------------------------------------------------------
    # Objects
    # -------------------------------------------------------------------------

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(
            pos=(0.60, -0.12, 0.05),
        ),

        "blue_cylinder": BOTTLE_INST_CFG(
            pos=(0.60, 0.12, 0.05),
        ),

        "table": TABLE_SEATTLE_INST_Cfg(
            pos=(0.50, 0.00, 0.00),
        ),
    }

    # -------------------------------------------------------------------------
    # Robot
    # -------------------------------------------------------------------------

    articulated_assets = {
        "robot": ROBOT_CFG(
            pos=(0.0, 0.0, 0.0),
        ),
    }

    # -------------------------------------------------------------------------
    # Cameras
    # -------------------------------------------------------------------------

    sensors = {
        # =====================================================================
        # Fixed external camera
        # =====================================================================
        "front_camera": CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_camera",
            update_period=0.0,
            update_latest_camera_pose=True,
            height=256,
            width=256,
            data_types=["rgb"],

            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),

            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        ),

        # =====================================================================
        # Wrist camera
        #
        # Both robot configs are expected to retain panda_hand.
        #
        # MatterixBaseEnv.setup_scene() appends the sensor name to the
        # configured prim path, so the final camera prim is approximately:
        #
        # .../Articulations_robot/panda_hand/camera_wrist_camera
        # =====================================================================
        "wrist_camera": CameraCfg(
            prim_path=(
                "{ENV_REGEX_NS}/"
                "Articulations_robot/"
                f"{WRIST_CAMERA_PARENT}/"
                "camera"
            ),
            update_period=0.0,
            update_latest_camera_pose=True,
            height=256,
            width=256,
            data_types=["rgb"],

            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.02, 5.0),
            ),

            offset=CameraCfg.OffsetCfg(
                pos=WRIST_CAMERA_POS,
                rot=WRIST_CAMERA_ROT,
                convention="ros",
            ),
        ),
    }

    # -------------------------------------------------------------------------
    # Gripper observation joints
    # -------------------------------------------------------------------------

    gripper_joint_names = GRIPPER_JOINT_NAMES

    # -------------------------------------------------------------------------
    # Observations / Events
    # -------------------------------------------------------------------------

    observations = ObservationManagerCfg()
    events = EventCfg()

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    record_path = None

    # -------------------------------------------------------------------------
    # Workflows
    # -------------------------------------------------------------------------

    workflows = {
        "pickup_beaker": PickObjectCfg(
            description="Pick up the beaker",
            agent_assets="robot",
            object="beaker",
            action_space_info=ACTION_SPACE_INFO,
        ),
    }