"""Shared simulator/imagination velocity task and command sampling."""
import math
import torch
from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse, yaw_quat, projected_gravity

RESAMPLE_STEPS = 250


def resample(metadata, ids):
    """Metadata columns: held vx,vy,wz, reference-source flag, remaining steps."""
    n = ids.numel()
    low = metadata.new_tensor([-1.5, -1.32, -3.35])
    high = metadata.new_tensor([2., 1.32, 3.35])
    command = low + torch.rand(n, 3, device=metadata.device) * (high-low)
    command[torch.rand(n, device=metadata.device) < .1] = 0
    metadata[ids, :3] = command
    metadata[ids, 3] = (torch.rand(n, device=metadata.device) < .5).float()
    metadata[ids, 4] = RESAMPLE_STEPS


def advance(metadata):
    metadata = metadata.clone()
    metadata[:, 4] -= 1
    resample(metadata, (metadata[:, 4] <= 0).nonzero().flatten())
    return metadata


def command_at(metadata, reference):
    return torch.where(metadata[:, 3:4].bool(), reference, metadata[:, :3])


def termination_features(s):
    # Upright = +1, horizontal = 0, upside-down = -1.
    up = -projected_gravity(s[..., 3:7])[..., 2]
    return torch.stack([s[..., 2], up], -1)


def terminated(s):
    f = termination_features(s)
    return (f[..., 0] < .35) | (f[..., 1] < math.cos(math.radians(73)))


def reward_terms(root_quat, root_lin_w, root_ang_w, torso_quat, torso_ang_w, q, limits, action, prev_action, command):
    """Reward rates, caller multiplies their sum by control dt exactly once."""
    v = quat_apply_inverse(yaw_quat(root_quat), root_lin_w)
    gravity = projected_gravity(torso_quat)
    # sin(roll)^2, independent of forward pitch except at the Euler singularity.
    lateral = gravity[..., 1].square() / (gravity[..., 1].square()+gravity[..., 2].square()).clamp_min(1e-8)
    torso_ang_b = quat_apply_inverse(torso_quat, torso_ang_w)
    return {
        'linear': 2*torch.exp(-(v[..., :2]-command[..., :2]).square().sum(-1)/.5**2),
        'yaw': 2*torch.exp(-(root_ang_w[..., 2]-command[..., 2]).square()/.5),
        'action_rate': -.3*(action-prev_action).square().sum(-1),
        'joint_limit': -2*((q-limits[..., 1]).clamp_min(0)+(limits[..., 0]-q).clamp_min(0)).sum(-1),
        'lateral_tilt': -4*lateral,
        'torso_angular': -.05*torso_ang_b[..., :2].square().sum(-1),
    }


def imagined_reward(wm, s, a, prev, command):
    b = wm.heads.robot_bodies(s)
    i = wm.heads.anchor_index  # torso_link
    terms = reward_terms(s[..., 3:7], s[..., 36:39], quat_apply(s[..., 3:7], s[..., 39:42]),
                         b['quat'][..., i, :], b['ang_vel'][..., i, :], s[..., 7:36],
                         wm.heads.soft_limits, a, prev, command)
    return sum(terms.values()) * wm.spec.cfg.dt


def motion_only(env, func, kwargs):
    cmd = env.command_manager.get_term('motion')
    return func(env, **kwargs) * (~cmd.velocity_mask)


def sim_reward(env):
    cmd = env.command_manager.get_term('motion')
    robot = cmd.robot.data
    ri, ti = cmd.robot_root_body_index, cmd.robot_anchor_body_index
    terms = reward_terms(robot.body_link_quat_w[:, ri], robot.body_link_lin_vel_w[:, ri], robot.body_link_ang_vel_w[:, ri],
                         robot.body_link_quat_w[:, ti], robot.body_link_ang_vel_w[:, ti], robot.joint_pos,
                         robot.soft_joint_pos_limits, env.action_manager.action, env.action_manager.prev_action, cmd.velocity_command)
    stats = getattr(cmd, "velocity_stats", None)
    if stats is not None:
        stats.update(robot.body_link_quat_w[:, ri], robot.body_link_lin_vel_w[:, ri], robot.body_link_ang_vel_w[:, ri],
                     cmd.velocity_command, cmd.velocity_metadata, cmd.velocity_mask, env.reset_terminated)
    return sum(terms.values()) * cmd.velocity_mask


def sim_terminated(env):
    cmd = env.command_manager.get_term('motion')
    height = cmd.robot_root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    up = -projected_gravity(cmd.robot_root_quat_w)[:, 2]
    return ((height < .35) | (up < math.cos(math.radians(73)))) & cmd.velocity_mask
