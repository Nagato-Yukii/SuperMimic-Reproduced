"""Fuse one differentiable velocity imagination step, including torso FK reward."""
import torch
from supermimic.velocity.task import imagined_reward
from supermimic.world_model.heads import analytic_observation


class VelocityStep:
    def __init__(self, wm, default_joint_pos, compile_mode=None):
        self.wm, self.q0 = wm, default_joint_pos
        self._fn = self._step
        if compile_mode:
            self._fn = torch.compile(self._step, mode=None if compile_mode == 'default' else compile_mode, dynamic=False)

    def _step(self, s, action, prev_action, xi, command):
        nxt = self.wm.step(s, action, xi)
        reward = imagined_reward(self.wm, nxt, action, prev_action, command)
        done = torch.sigmoid(self.wm.velocity_termination_logits(nxt))
        observation = analytic_observation(nxt, action, xi, self.q0)
        return nxt, reward, done, observation

    def __call__(self, s, action, prev_action, xi, command):
        return self._fn(s, action, prev_action, xi, command)
