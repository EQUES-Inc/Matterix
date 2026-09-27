#!/usr/bin/env python3
"""
MATTERIX TCP server for VLA-JEPA zero-shot evaluation.

Run this script in the `matterix` conda environment.

Responsibilities:
    - start Isaac Sim / Isaac Lab / MATTERIX
    - create the FrankaBeakerLift environment
    - receive TCP commands from eval_matterix.py
    - return RGB observations
    - receive 7D VLA-JEPA delta actions:
        [dx, dy, dz, dRx, dRy, dRz, gripper]
    - convert them into MATTERIX 8D absolute IK actions:
        [x, y, z, qw, qx, qy, qz, gripper]

Recommended location:
    ~/Matterix/scripts/matterix_server.py
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import signal
import socket
import struct
import threading
import traceback
from typing import Any

import imageio.v2 as imageio
import numpy as np

shutdown_event = threading.Event()
response_sequence = 0


# ===========================================================================
# CLI arguments before AppLauncher
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MATTERIX TCP server for VLA-JEPA evaluation."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Exact Gym task ID for FrankaBeakerLift.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host/IP on which the MATTERIX TCP server listens.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="TCP port for MATTERIX server.",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of MATTERIX environments. Use 1 for VLA-JEPA evaluation.",
    )

    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cuda:0",
    #     help="Isaac Lab simulation device.",
    # )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base environment seed.",
    )

    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help="Scale applied to VLA-JEPA translation deltas.",
    )

    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="Scale applied to VLA-JEPA axis-angle rotation deltas.",
    )

    parser.add_argument(
        "--rotation-frame",
        type=str,
        choices=["world", "local"],
        default="world",
        help=(
            "Quaternion composition convention. "
            "'world': q_target = q_delta * q_current. "
            "'local': q_target = q_current * q_delta."
        ),
    )

    parser.add_argument(
        "--lift-threshold",
        type=float,
        default=0.10,
        help="Required vertical beaker displacement in meters for success.",
    )

    parser.add_argument(
        "--invert-gripper",
        action="store_true",
        help="Invert the gripper command sign before sending it to MATTERIX.",
    )

    parser.add_argument(
        "--primary-camera-name",
        type=str,
        default="front_camera",
        help="Scene sensor name for the fixed external RGB camera.",
    )

    parser.add_argument(
        "--wrist-camera-name",
        type=str,
        default="wrist_camera",
        help="Scene sensor name for the wrist RGB camera.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser


parser = build_parser()


# ===========================================================================
# Isaac Lab AppLauncher
# ===========================================================================

from isaaclab.app import AppLauncher


AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()


def debug_print(*values, **kwargs) -> None:
    """Print verbose diagnostics only when --debug is enabled."""
    if args.debug:
        kwargs.setdefault("flush", True)
        print(*values, **kwargs)


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ===========================================================================
# Imports that require Isaac Sim to be running
# ===========================================================================

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import matterix_tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_mul


# ===========================================================================
# TCP protocol
# ===========================================================================


def send_message(
    sock: socket.socket,
    obj: Any,
) -> None:
    """
    Serialize a Python object with pickle and send it with an 8-byte
    big-endian payload-size header.
    """
    payload = pickle.dumps(
        obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    header = struct.pack(
        "!Q",
        len(payload),
    )

    sock.sendall(header + payload)


def recv_exact(
    sock: socket.socket,
    n: int,
) -> bytes:
    """
    Receive exactly n bytes.
    """
    chunks: list[bytes] = []
    remaining = n

    while remaining > 0:
        if shutdown_event.is_set():
            raise KeyboardInterrupt("Shutdown requested.")

        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            continue

        if not chunk:
            raise ConnectionError(
                "Socket closed while receiving data."
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def recv_message(
    sock: socket.socket,
) -> Any:
    """
    Receive one size-prefixed pickle message.
    """
    header = recv_exact(sock, 8)

    (length,) = struct.unpack(
        "!Q",
        header,
    )

    payload = recv_exact(
        sock,
        length,
    )

    return pickle.loads(payload)


# ===========================================================================
# Utility helpers
# ===========================================================================


def get_base_env(env):
    """
    Return the unwrapped MATTERIX environment.
    """
    return env.unwrapped


def print_scene_keys(env) -> None:
    """
    Print available InteractiveScene entries.
    """
    base_env = get_base_env(env)

    try:
        keys = list(base_env.scene.keys())
    except Exception:
        keys = []

    debug_print(
        f"[DEBUG] scene keys: {keys}",
        flush=True,
    )


# ===========================================================================
# Camera
# ===========================================================================


def get_camera_rgb(
    env,
    camera_name: str,
) -> np.ndarray:
    """
    Read RGB data directly from an Isaac Lab CameraCfg sensor.

    Expected source:
        env.unwrapped.scene[camera_name].data.output["rgb"]

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8.
    """
    base_env = get_base_env(env)

    debug_print(
        f"[CAMERA] Reading camera: {camera_name}",
        flush=True,
    )

    print_scene_keys(env)

    try:
        camera = base_env.scene[camera_name]
    except Exception as exc:
        raise KeyError(
            f"Camera '{camera_name}' was not found in the scene."
        ) from exc

    output = camera.data.output

    debug_print(
        f"[CAMERA] output keys: {list(output.keys())}",
        flush=True,
    )

    if "rgb" not in output:
        raise KeyError(
            f"Camera '{camera_name}' has no 'rgb' output. "
            f"Available outputs: {list(output.keys())}"
        )

    rgb = output["rgb"]

    debug_print(
        f"[CAMERA] raw rgb shape: {getattr(rgb, 'shape', None)}",
        flush=True,
    )

    debug_print(
        f"[CAMERA] raw rgb dtype: {getattr(rgb, 'dtype', None)}",
        flush=True,
    )

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()

    rgb = np.asarray(rgb)

    # Expected batched layout:
    #   (num_envs, H, W, C)
    if rgb.ndim == 4:
        rgb = rgb[0]

    if rgb.ndim != 3:
        raise ValueError(
            f"Unexpected RGB shape: {rgb.shape}. "
            "Expected HWC or NHWC."
        )

    # RGBA -> RGB
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if rgb.shape[-1] != 3:
        raise ValueError(
            f"Unexpected number of RGB channels: {rgb.shape}"
        )

    # Handle float images if necessary.
    if np.issubdtype(rgb.dtype, np.floating):
        max_value = float(np.max(rgb)) if rgb.size > 0 else 0.0

        if max_value <= 1.0 + 1e-6:
            rgb = rgb * 255.0

    rgb = np.clip(
        rgb,
        0,
        255,
    ).astype(np.uint8)

    rgb = np.ascontiguousarray(rgb)

    debug_print(
        f"[CAMERA] final rgb shape: {rgb.shape}",
        flush=True,
    )

    debug_print(
        f"[CAMERA] final rgb dtype: {rgb.dtype}",
        flush=True,
    )

    return rgb


def configure_front_camera(env) -> None:
    base_env = get_base_env(env)
    camera = base_env.scene[args.primary_camera_name]

    eyes = torch.tensor(
        [[0.90, -0.15, 0.58]], #oblique front camera
        # [[0.90, 0.00, 0.58]],    #frontal front camera
        device=base_env.device,
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [[0.50, 0.02, 0.16]],
        device=base_env.device,
        dtype=torch.float32,
    )

    camera.set_world_poses_from_view(
        eyes=eyes,
        targets=targets,
        env_ids=[0],
    )

    base_env.sim.render()
    camera.update(dt=base_env.physics_dt)
    

# ===========================================================================
# Beaker state
# ===========================================================================


def get_beaker_height(env) -> float:
    """
    Get the current world-frame Z coordinate of the beaker.
    """
    base_env = get_base_env(env)

    debug_print(
        "[STATE] Reading beaker height...",
        flush=True,
    )

    try:
        beaker = base_env.scene["beaker"]
    except Exception as exc:
        print_scene_keys(env)

        raise KeyError(
            "Scene object 'beaker' was not found."
        ) from exc

    root_pos_w = beaker.data.root_pos_w

    debug_print(
        f"[STATE] beaker root_pos_w: {root_pos_w}",
        flush=True,
    )

    z = float(
        root_pos_w[0, 2].item()
    )

    debug_print(
        f"[STATE] beaker z: {z:.6f}",
        flush=True,
    )

    return z

def get_beaker_position(env) -> np.ndarray:
    base_env = env.unwrapped
    beaker = base_env.scene["beaker"]

    return (
        beaker.data.root_pos_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def get_object_position(env, object_name: str) -> np.ndarray:
    """Return one scene object's world position as float32 xyz."""
    base_env = get_base_env(env)
    try:
        obj = base_env.scene[object_name]
    except Exception as exc:
        print_scene_keys(env)
        raise KeyError(f"Scene object '{object_name}' was not found.") from exc

    return (
        obj.data.root_pos_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def get_object_height(env, object_name: str) -> float:
    return float(get_object_position(env, object_name)[2])


def set_target_object_layout(env, reset_seed: int) -> None:
    base_env = get_base_env(env)

    beaker = base_env.scene["beaker"]
    cylinder = base_env.scene["blue_cylinder"]

    # 乱数はreset_seed依存にして再現可能にする
    rng = np.random.default_rng(reset_seed)

    swap = False #True or Falseで指定
    # swap = bool(rng.integers(0, 2))　# 左右を50/50でswap
    # swap = bool(reset_seed % 2) # 偶奇でわける
    

    left_y = -0.12
    right_y = 0.12

    # 小さなjitter
    beaker_x_jitter = rng.uniform(-0.02, 0.02)
    beaker_y_jitter = rng.uniform(-0.02, 0.02)

    cylinder_x_jitter = rng.uniform(-0.02, 0.02)
    cylinder_y_jitter = rng.uniform(-0.02, 0.02)

    if not swap:
        beaker_y = left_y
        cylinder_y = right_y
    else:
        beaker_y = right_y
        cylinder_y = left_y

    beaker_root_state = beaker.data.root_state_w.clone()
    cylinder_root_state = cylinder.data.root_state_w.clone()

    # xyz
    beaker_root_state[0, 0] = 0.60 + beaker_x_jitter
    beaker_root_state[0, 1] = beaker_y + beaker_y_jitter
    beaker_root_state[0, 2] = 0.05

    cylinder_root_state[0, 0] = 0.60 + cylinder_x_jitter
    cylinder_root_state[0, 1] = cylinder_y + cylinder_y_jitter
    cylinder_root_state[0, 2] = 0.05

    # linear/angular velocity = 0
    beaker_root_state[:, 7:13] = 0.0
    cylinder_root_state[:, 7:13] = 0.0

    beaker.write_root_state_to_sim(beaker_root_state)
    cylinder.write_root_state_to_sim(cylinder_root_state)

    base_env.sim.forward()

    print(
        "[LAYOUT]",
        f"seed={reset_seed}",
        f"swap={swap}",
        f"beaker={get_object_position(env, 'beaker')}",
        f"blue_cylinder={get_object_position(env, 'blue_cylinder')}",
        flush=True,
    )

# ===========================================================================
# Quaternion / rotation utilities
# ===========================================================================


def axis_angle_to_quaternion_wxyz(
    axis_angle: torch.Tensor,
) -> torch.Tensor:
    """
    Convert an axis-angle rotation vector into a quaternion in WXYZ order.

    Input:
        shape (..., 3)

    Output:
        shape (..., 4), WXYZ
    """
    angle = torch.linalg.norm(
        axis_angle,
        dim=-1,
        keepdim=True,
    )

    half_angle = 0.5 * angle

    eps = 1e-8

    axis = axis_angle / torch.clamp(
        angle,
        min=eps,
    )

    xyz = axis * torch.sin(half_angle)

    w = torch.cos(half_angle)

    quat = torch.cat(
        [
            w,
            xyz,
        ],
        dim=-1,
    )

    # For extremely small angles, use identity quaternion.
    small = angle.squeeze(-1) < eps

    if torch.any(small):
        quat = quat.clone()

        quat[small] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=quat.device,
            dtype=quat.dtype,
        )

    return quat


def quaternion_wxyz_to_axis_angle(
    quat: torch.Tensor,
) -> torch.Tensor:
    """
    Convert WXYZ quaternion into an axis-angle vector.

    Input:
        shape (..., 4)

    Output:
        shape (..., 3)
    """
    quat = quat / torch.clamp(
        torch.linalg.norm(
            quat,
            dim=-1,
            keepdim=True,
        ),
        min=1e-8,
    )

    w = torch.clamp(
        quat[..., 0],
        -1.0,
        1.0,
    )

    xyz = quat[..., 1:]

    sin_half = torch.linalg.norm(
        xyz,
        dim=-1,
        keepdim=True,
    )

    angle = 2.0 * torch.atan2(
        sin_half.squeeze(-1),
        w,
    )

    eps = 1e-8

    axis = xyz / torch.clamp(
        sin_half,
        min=eps,
    )

    axis_angle = axis * angle.unsqueeze(-1)

    small = sin_half.squeeze(-1) < eps

    if torch.any(small):
        axis_angle = axis_angle.clone()
        axis_angle[small] = 0.0

    return axis_angle


# ===========================================================================
# Robot state
# ===========================================================================


def get_robot_control_pose(
    env,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the current end-effector control frame pose.

    L-link + Robotiq:
        body_name = "virtual_eef_link"
        body_offset = none

    Standard Panda:
        body_name = "panda_hand"
        body_offset.pos = (0.0, 0.0, 0.107)

    Returns:
        position: shape (1, 3)
        quaternion WXYZ: shape (1, 4)
    """
    base_env = get_base_env(env)
    robot = base_env.scene["robot"]

    body_names = robot.data.body_names

    if "virtual_eef_link" in body_names:
        control_body_name = "virtual_eef_link"
        offset_z = 0.0

    elif "panda_hand" in body_names:
        control_body_name = "panda_hand"
        offset_z = 0.107

    else:
        raise KeyError(
            "No supported robot control body found. "
            f"Available bodies: {body_names}"
        )

    body_index = body_names.index(control_body_name)

    body_pos_w = robot.data.body_pos_w[
        :,
        body_index,
        :,
    ]

    body_quat_w = robot.data.body_quat_w[
        :,
        body_index,
        :,
    ]

    if offset_z != 0.0:
        offset = torch.tensor(
            [[0.0, 0.0, offset_z]],
            device=body_pos_w.device,
            dtype=body_pos_w.dtype,
        )

        offset_w = quat_apply(
            body_quat_w,
            offset,
        )

        control_pos_w = body_pos_w + offset_w
    else:
        control_pos_w = body_pos_w

    control_quat_w = body_quat_w

    return (
        control_pos_w,
        control_quat_w,
    )


def get_robot_state(
    env,
) -> np.ndarray:
    """
    Build a LIBERO-like state vector:

        EE position:   3
        EE axis-angle: 3
        gripper qpos:  2

    Total:
        8 dimensions
    """
    base_env = get_base_env(env)

    control_pos_w, control_quat_w = (
        get_robot_control_pose(env)
    )

    axis_angle = (
        quaternion_wxyz_to_axis_angle(
            control_quat_w
        )
    )

    robot = base_env.scene["robot"]

    joint_names = list(
        robot.data.joint_names
    )

    # ---------------------------------------------------------
    # Detect gripper joints
    # ---------------------------------------------------------
    robotiq_gripper_joint_names = [
        "robotiq_85_left_knuckle_joint",
        "robotiq_85_right_knuckle_joint",
    ]

    panda_gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    if all(
        name in joint_names
        for name in robotiq_gripper_joint_names
    ):
        gripper_joint_names = (
            robotiq_gripper_joint_names
        )

    elif all(
        name in joint_names
        for name in panda_gripper_joint_names
    ):
        gripper_joint_names = (
            panda_gripper_joint_names
        )

    else:
        raise KeyError(
            "No supported gripper joints found. "
            f"Available joints: {joint_names}"
        )

    gripper_indices = [
        joint_names.index(name)
        for name in gripper_joint_names
    ]

    gripper_pos = robot.data.joint_pos[
        :,
        gripper_indices,
    ]

    debug_print(
        "[STATE DEBUG] control_pos_w:",
        control_pos_w.detach().cpu().numpy(),
        flush=True,
    )

    debug_print(
        "[STATE DEBUG] axis_angle:",
        axis_angle.detach().cpu().numpy(),
        flush=True,
    )

    debug_print(
        "[STATE DEBUG] gripper_indices:",
        gripper_indices,
        flush=True,
    )

    debug_print(
        "[STATE DEBUG] gripper_joint_names:",
        gripper_joint_names,
        flush=True,
    )

    debug_print(
        "[STATE DEBUG] gripper_pos:",
        gripper_pos.detach().cpu().numpy(),
        flush=True,
    )

    state = torch.cat(
        [
            control_pos_w,
            axis_angle,
            gripper_pos,
        ],
        dim=-1,
    )

    result = (
        state[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    debug_print(
        f"[STATE] robot state shape: {result.shape}",
        flush=True,
    )

    debug_print(
        f"[STATE] robot state: {result}",
        flush=True,
    )

    return result

def open_gripper_after_reset(
    env,
    num_steps: int = 20,
) -> None:
    control_pos, control_quat = get_robot_control_pose(env)

    open_action = torch.cat(
        [
            control_pos,
            control_quat,
            torch.ones(
                (1, 1),
                device=control_pos.device,
                dtype=torch.float32,
            ),
        ],
        dim=-1,
    )

    for _ in range(num_steps):
        env.step(open_action)

    state = get_robot_state(env)

    debug_print(
        "[RESET DEBUG] state after opening gripper:",
        state,
        flush=True,
    )

# ===========================================================================
# VLA-JEPA 7D delta -> MATTERIX 8D absolute IK action
# ===========================================================================


def convert_vla_action_to_matterix(
    env,
    action_7d: np.ndarray,
) -> torch.Tensor:
    """
    Convert:

        VLA-JEPA:
            [dx, dy, dz, dRx, dRy, dRz, gripper]

    into:

        MATTERIX:
            [x, y, z, qw, qx, qy, qz, gripper]
    """
    action_7d = np.asarray(
        action_7d,
        dtype=np.float32,
    ).reshape(-1)

    if action_7d.size != 7:
        raise ValueError(
            f"Expected 7D VLA-JEPA action, got shape {action_7d.shape}"
        )

    base_env = get_base_env(env)

    device = base_env.device

    current_pos, current_quat = (
        get_robot_control_pose(env)
    )

    raw_position_delta = torch.tensor(
        action_7d[:3],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    # VLA-JEPA/LIBERO action frame -> MATTERIX world frame
    position_axis_sign = torch.tensor(
        [[1.0, 1.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    position_delta = (
        raw_position_delta
        * position_axis_sign
    )

    rotation_delta = torch.tensor(
        action_7d[3:6],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    position_delta = (
        position_delta
        * args.position_scale
    )

    rotation_delta = (
        rotation_delta
        * args.rotation_scale
    )

    target_pos = (
        current_pos
        + position_delta
    )

    delta_quat = (
        axis_angle_to_quaternion_wxyz(
            rotation_delta
        )
    )

    if args.rotation_frame == "world":
        target_quat = quat_mul(
            delta_quat,
            current_quat,
        )
    else:
        target_quat = quat_mul(
            current_quat,
            delta_quat,
        )

    target_quat = target_quat / torch.clamp(
        torch.linalg.norm(
            target_quat,
            dim=-1,
            keepdim=True,
        ),
        min=1e-8,
    )

    gripper = float(
        action_7d[6]
    )

    if args.invert_gripper:
        gripper *= -1.0

    gripper_tensor = torch.tensor(
        [[gripper]],
        device=device,
        dtype=torch.float32,
    )

    matterix_action = torch.cat(
        [
            target_pos,
            target_quat,
            gripper_tensor,
        ],
        dim=-1,
    )

    if matterix_action.shape != (1, 8):
        raise RuntimeError(
            "Unexpected MATTERIX action shape: "
            f"{tuple(matterix_action.shape)}"
        )

    debug_print(
        "[ACTION] VLA 7D:",
        action_7d,
        flush=True,
    )

    debug_print(
        "[ACTION] MATTERIX 8D:",
        matterix_action.detach().cpu().numpy(),
        flush=True,
    )


    debug_print(
        "[ACTION DEBUG] current_pos:",
        current_pos.detach().cpu().numpy(),
        flush=True,
    )

    debug_print(
        "[ACTION DEBUG] raw position_delta:",
        torch.tensor(
            action_7d[:3]
        ).numpy(),
        flush=True,
    )

    debug_print(
        "[ACTION DEBUG] raw VLA position_delta:",
        raw_position_delta.detach().cpu().numpy(),
        flush=True,
    )

    debug_print(
        "[ACTION DEBUG] mapped position_delta before scale:",
        (
            raw_position_delta
            * position_axis_sign
        ).detach().cpu().numpy(),
        flush=True,
    )

    debug_print(
        "[ACTION DEBUG] scaled position_delta:",
        position_delta.detach().cpu().numpy(),
        flush=True,
    )

    debug_print(
        "[ACTION DEBUG] target_pos:",
        target_pos.detach().cpu().numpy(),
        flush=True,
    )
    return matterix_action


# ===========================================================================
# Gym step compatibility
# ===========================================================================


def parse_step_result(
    result,
):
    """
    Handle Gymnasium's 5-item step return.
    """
    if not isinstance(result, tuple):
        raise TypeError(
            f"Unexpected env.step() return type: {type(result)}"
        )

    if len(result) != 5:
        raise ValueError(
            f"Expected 5 values from env.step(), got {len(result)}"
        )

    obs, reward, terminated, truncated, info = result

    terminated_flag = bool(
        np.asarray(
            terminated.detach().cpu().numpy()
            if isinstance(terminated, torch.Tensor)
            else terminated
        ).reshape(-1)[0]
    )

    truncated_flag = bool(
        np.asarray(
            truncated.detach().cpu().numpy()
            if isinstance(truncated, torch.Tensor)
            else truncated
        ).reshape(-1)[0]
    )

    done = (
        terminated_flag
        or truncated_flag
    )

    return (
        obs,
        reward,
        done,
        info,
    )


# ===========================================================================
# Response construction
# ===========================================================================


def make_response(
    env,
    initial_beaker_z: float,
    initial_blue_cylinder_z: float,
    done: bool,
) -> dict[str, Any]:
    """
    Create one observation packet and attach an immutable response ID/hash.

    The exact wrist array placed in response["wrist_image"] is also saved
    with the same response ID. This makes server/client frame comparison
    unambiguous.
    """
    global response_sequence

    response_id = response_sequence
    response_sequence += 1

    debug_print(
        f"[RESPONSE] Creating response_id={response_id}...",
        flush=True,
    )

    primary_image = np.array(
        get_camera_rgb(
            env,
            camera_name=args.primary_camera_name,
        ),
        dtype=np.uint8,
        order="C",
        copy=True,
    )

    wrist_image = np.array(
        get_camera_rgb(
            env,
            camera_name=args.wrist_camera_name,
        ),
        dtype=np.uint8,
        order="C",
        copy=True,
    )

    primary_image = np.ascontiguousarray(primary_image)
    wrist_image = np.ascontiguousarray(wrist_image)

    primary_sha256 = hashlib.sha256(
        primary_image.tobytes()
    ).hexdigest()

    wrist_sha256 = hashlib.sha256(
        wrist_image.tobytes()
    ).hexdigest()

    primary_path = (
        f"./matterix_response_{response_id:06d}_primary.png"
    )
    wrist_path = (
        f"./matterix_response_{response_id:06d}_wrist.png"
    )

    # imageio.imwrite(primary_path, primary_image)
    # imageio.imwrite(wrist_path, wrist_image)

    debug_print(
        "[RESPONSE][PRIMARY]",
        f"id={response_id}",
        f"path={primary_path}",
        f"shape={primary_image.shape}",
        f"dtype={primary_image.dtype}",
        f"min={int(primary_image.min())}",
        f"max={int(primary_image.max())}",
        f"mean={float(primary_image.mean()):.3f}",
        f"sha256={primary_sha256}",
        flush=True,
    )

    debug_print(
        "[RESPONSE][WRIST]",
        f"id={response_id}",
        f"path={wrist_path}",
        f"shape={wrist_image.shape}",
        f"dtype={wrist_image.dtype}",
        f"min={int(wrist_image.min())}",
        f"max={int(wrist_image.max())}",
        f"mean={float(wrist_image.mean()):.3f}",
        f"sha256={wrist_sha256}",
        flush=True,
    )

    current_beaker_z = get_beaker_height(env)
    beaker_lift = current_beaker_z - initial_beaker_z
    beaker_success = bool(beaker_lift >= args.lift_threshold)

    blue_cylinder_position = get_object_position(env, "blue_cylinder")
    current_blue_cylinder_z = float(blue_cylinder_position[2])
    blue_cylinder_lift = current_blue_cylinder_z - initial_blue_cylinder_z
    blue_cylinder_success = bool(blue_cylinder_lift >= args.lift_threshold)
 
    # Keep legacy success/done semantics for backward compatibility.
    success = beaker_success

    try:
        state = get_robot_state(env)
    except Exception as exc:
        print(
            f"[WARNING] Failed to get robot state: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        state = None

    response = {
        "response_id": response_id,
        "primary_sha256": primary_sha256,
        "wrist_sha256": wrist_sha256,
        "primary_image": primary_image,
        "wrist_image": wrist_image,
        "primary_camera_name": args.primary_camera_name,
        "wrist_camera_name": args.wrist_camera_name,
        "state": state,
        "beaker_position": get_beaker_position(env),
        "beaker_z": current_beaker_z,
        "beaker_lift": beaker_lift,
        "beaker_success": beaker_success,
        "blue_cylinder_position": blue_cylinder_position,
        "blue_cylinder_z": current_blue_cylinder_z,
        "blue_cylinder_lift": blue_cylinder_lift,
        "blue_cylinder_success": blue_cylinder_success,
        "success": success,
        "done": bool(done or beaker_success or blue_cylinder_success),
    }

    debug_print(
        f"[RESPONSE] response_id={response_id} "
        f"keys={list(response.keys())}",
        flush=True,
    )

    return response


# ===========================================================================
# Client handling
# ===========================================================================


def handle_client(
    conn: socket.socket,
    address,
    env,
) -> None:
    """
    Handle a single eval_matterix.py TCP client.
    """
    print(
        f"[SERVER] Client connected from {address}",
        flush=True,
    )

    initial_beaker_z: float | None = None
    initial_blue_cylinder_z: float | None = None
    conn.settimeout(1.0)

    try:
        while not shutdown_event.is_set():
            print(
                "[SERVER] Waiting for request...",
                flush=True,
            )

            request = recv_message(conn)

            print(
                f"[SERVER] Request received: {request}",
                flush=True,
            )

            if not isinstance(request, dict):
                raise TypeError(
                    "Expected request to be a dict, "
                    f"got {type(request)}"
                )

            cmd = request.get("cmd")

            print(
                f"[SERVER] Command: {cmd}",
                flush=True,
            )

            # ===============================================================
            # RESET
            # ===============================================================

            if cmd == "reset":
                episode_idx = int(
                    request.get(
                        "episode_idx",
                        0,
                    )
                )

                print(
                    f"[SERVER][RESET] Episode index: {episode_idx}",
                    flush=True,
                )

                reset_seed = (
                    args.seed
                    + episode_idx
                )

                print(
                    f"[SERVER][RESET] Calling env.reset(seed={reset_seed})...",
                    flush=True,
                )

                reset_result = env.reset(
                    seed=reset_seed
                )

                set_target_object_layout(
                    env,
                    reset_seed=reset_seed,
                )

                print(
                    "[SERVER][RESET] env.reset() completed",
                    flush=True,
                )

                open_gripper_after_reset(
                    env,
                    num_steps=20,
                )

                configure_front_camera(env)

                print_scene_keys(env)

                initial_beaker_z = (
                    get_beaker_height(env)
                )
                initial_blue_cylinder_z = get_object_height(
                    env, "blue_cylinder"
                )

                print(
                    "[SERVER][RESET] "
                    f"initial_beaker_z={initial_beaker_z:.6f} "
                    f"initial_blue_cylinder_z={initial_blue_cylinder_z:.6f}",
                    flush=True,
                )

                print(
                    "[SERVER][RESET] Creating response...",
                    flush=True,
                )

                response = make_response(
                    env=env,
                    initial_beaker_z=initial_beaker_z,
                    initial_blue_cylinder_z=initial_blue_cylinder_z,
                    done=False,
                )

                print(
                    "[SERVER][RESET] Sending response...",
                    flush=True,
                )

                send_message(
                    conn,
                    response,
                )

                print(
                    "[SERVER][RESET] Response sent successfully",
                    flush=True,
                )

            # ===============================================================
            # STEP
            # ===============================================================

            elif cmd == "step":
                if initial_beaker_z is None or initial_blue_cylinder_z is None:
                    raise RuntimeError(
                        "Received 'step' before 'reset'."
                    )

                if "action" not in request:
                    raise KeyError(
                        "Step request has no 'action' field."
                    )

                action_7d = np.asarray(
                    request["action"],
                    dtype=np.float32,
                ).reshape(-1)

                print(
                    f"[SERVER][STEP] Received action: {action_7d}",
                    flush=True,
                )

                action_8d = (
                    convert_vla_action_to_matterix(
                        env,
                        action_7d,
                    )
                )
                
                actual_pos_before, _ = get_robot_control_pose(env)

                debug_print(
                    "[ACTION DEBUG] actual_pos_before:",
                    actual_pos_before.detach().cpu().numpy(),
                    flush=True,
                )

                print(
                    "[SERVER][STEP] Calling env.step()...",
                    flush=True,
                )

                step_result = env.step(
                    action_8d
                )

                print(
                    "[SERVER][STEP] env.step() completed",
                    flush=True,
                )

                actual_pos_after, _ = get_robot_control_pose(env)

                debug_print(
                    "[ACTION DEBUG] actual_pos_after:",
                    actual_pos_after.detach().cpu().numpy(),
                    flush=True,
                )

                debug_print(
                    "[ACTION DEBUG] actual displacement:",
                    (
                        actual_pos_after
                        - actual_pos_before
                    ).detach().cpu().numpy(),
                    flush=True,
                )

                obs, reward, done, info = parse_step_result(step_result)

                print(
                    f"[SERVER][STEP] done={done}",
                    flush=True,
                )

                response = make_response(
                    env=env,
                    initial_beaker_z=initial_beaker_z,
                    initial_blue_cylinder_z=initial_blue_cylinder_z,
                    done=done,
                )

                print(
                    "[SERVER][STEP] Sending response...",
                    flush=True,
                )

                send_message(
                    conn,
                    response,
                )

                print(
                    "[SERVER][STEP] Response sent successfully",
                    flush=True,
                )

            # ===============================================================
            # CLOSE
            # ===============================================================

            elif cmd == "close":
                print(
                    "[SERVER] Close command received",
                    flush=True,
                )

                send_message(
                    conn,
                    {
                        "status": "closed",
                    },
                )

                print(
                    "[SERVER] Close response sent",
                    flush=True,
                )

                break

            else:
                raise ValueError(
                    f"Unknown command: {cmd}"
                )

    except KeyboardInterrupt:
        print(
            "[SERVER] Client handler stopping due to shutdown request.",
            flush=True,
        )

    except ConnectionError as exc:
        print(
            f"[SERVER] Client disconnected: {exc}",
            flush=True,
        )

    except Exception as exc:
        print(
            f"[SERVER ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        traceback.print_exc()

        # Try to report the server-side exception to the client.
        try:
            send_message(
                conn,
                {
                    "server_error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        except Exception:
            pass

    finally:
        print(
            "[SERVER] Closing client socket",
            flush=True,
        )

        try:
            conn.close()
        except Exception:
            pass


# ===========================================================================
# Graceful shutdown
# ===========================================================================


def handle_shutdown_signal(signum, frame) -> None:
    print(f"\n[MAIN] Shutdown signal received: {signum}", flush=True)
    shutdown_event.set()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    env = None
    server = None

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        print(
            "[MAIN] Parsing environment configuration...",
            flush=True,
        )

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
        )

        # Disable MATTERIX dataset recording for VLA-JEPA evaluation.
        env_cfg.record_path = None

        print(
            f"[MAIN] Task: {args.task}",
            flush=True,
        )

        print(
            f"[MAIN] Device: {args.device}",
            flush=True,
        )

        print(
            f"[MAIN] Number of environments: {args.num_envs}",
            flush=True,
        )

        print(
            "[MAIN] Creating MATTERIX environment...",
            flush=True,
        )

        env = gym.make(
            args.task,
            cfg=env_cfg,
        )

        wrist = env.unwrapped.scene["wrist_camera"]

        debug_print(
            "[WRIST DEBUG] prim_path:",
            wrist.cfg.prim_path,
            flush=True,
        )

        debug_print(
            "[WRIST DEBUG] offset:",
            wrist.cfg.offset,
            flush=True,
        )

        print(
            "[MAIN] Environment created successfully",
            flush=True,
        )

        print(
            f"[MAIN] observation_space: {env.observation_space}",
            flush=True,
        )

        print(
            f"[MAIN] action_space: {env.action_space}",
            flush=True,
        )

        debug_print(
            "[DEBUG] robot cfg prim_path:",
            env.unwrapped.scene["robot"].cfg.prim_path,
            flush=True,
        )

        if tuple(
            env.action_space.shape
        ) != (
            args.num_envs,
            8,
        ):
            raise RuntimeError(
                "Unexpected action space. "
                f"Expected ({args.num_envs}, 8), "
                f"got {env.action_space.shape}"
            )

        print_scene_keys(env)

        # -------------------------------------------------------------------
        # Initial reset / camera validation
        # -------------------------------------------------------------------

        print(
            "[MAIN] Performing initial environment reset...",
            flush=True,
        )

        reset_result = env.reset(
            seed=args.seed
        )

        set_target_object_layout(
            env,
            reset_seed=args.seed,
        )

        open_gripper_after_reset(
            env,
            num_steps=20,
        )

        configure_front_camera(env)

        print(
            "[MAIN] Testing primary RGB camera...",
            flush=True,
        )

        primary_rgb = get_camera_rgb(
            env,
            camera_name=args.primary_camera_name,
        )

        print(
            "[MAIN] primary rgb:",
            primary_rgb.shape,
            primary_rgb.dtype,
            flush=True,
        )

        print(
            "[MAIN] Testing wrist RGB camera...",
            flush=True,
        )

        wrist_rgb = get_camera_rgb(
            env,
            camera_name=args.wrist_camera_name,
        )

        print(
            "[MAIN] wrist rgb:",
            wrist_rgb.shape,
            wrist_rgb.dtype,
            flush=True,
        )

        
        imageio.imwrite(
            "./matterix_front_camera.png",
            primary_rgb,
        )

        imageio.imwrite(
            "./matterix_wrist_camera.png",
            wrist_rgb,
        )

        print(
            "[MAIN] Saved camera images:",
            "./matterix_front_camera.png",
            "./matterix_wrist_camera.png",
            flush=True,
        )

        # -------------------------------------------------------------------
        # TCP server
        # -------------------------------------------------------------------

        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.bind(
            (
                args.host,
                args.port,
            )
        )

        server.listen(1)
        server.settimeout(1.0)

        print(
            "============================================================",
            flush=True,
        )

        print(
            f"[SERVER] MATTERIX server listening on "
            f"{args.host}:{args.port}",
            flush=True,
        )

        print(
            "============================================================",
            flush=True,
        )

        # Keep accepting clients.
        #
        # This is useful during debugging because a failed eval_matterix.py
        # connection does not require restarting Isaac Sim every time.
        while not shutdown_event.is_set():
            print(
                "[SERVER] Waiting for client connection...",
                flush=True,
            )

            try:
                conn, address = server.accept()
            except socket.timeout:
                continue

            handle_client(
                conn=conn,
                address=address,
                env=env,
            )

            print(
                "[SERVER] Client disconnected. "
                "Returning to accept() loop.",
                flush=True,
            )

    except KeyboardInterrupt:
        print(
            "\n[MAIN] KeyboardInterrupt received",
            flush=True,
        )

    except Exception as exc:
        print(
            f"[MAIN ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        traceback.print_exc()

        raise

    finally:
        print(
            "[MAIN] Cleaning up...",
            flush=True,
        )

        shutdown_event.set()

        if server is not None:
            try:
                server.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass

            try:
                server.close()
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()

        try:
            simulation_app.close()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()