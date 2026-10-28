"""Velocity BPTT, reference-only latent alignment, and a separate GAE critic."""
from contextlib import contextmanager
import torch
from torch import nn
from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major, anchor_relative_obs
from supermimic.world_model.wm_trainer import reference_at
from supermimic.imagination.rollout import OBS_NOISE_HALF
from supermimic.imagination.objective import gae_targets
from supermimic.training.distributed import allreduce_grads
from supermimic.velocity.task import advance, command_at
from supermimic.velocity.compiled import VelocityStep


@contextmanager
def frozen(*modules):
    params = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in params:
            p.requires_grad_(flag)


class VelocityCritic(nn.Module):
    def __init__(self, hidden=(256,128)):
        super().__init__()
        # State, current command, simulation context, source flag / countdown.
        self.mlp = MLP(71+3+33+2, hidden, 1)

    def forward(self, feat, command, xi, metadata):
        context = torch.stack([metadata[..., 3], metadata[..., 4]/250], -1)
        return self.mlp(torch.cat([feat, command, xi, context], -1)).squeeze(-1)


def masked_alignment(z_velocity, z_motion, is_reference):
    per_sample = (z_velocity-z_motion.detach()).square().mean(-1)
    return (per_sample*is_reference).sum(), is_reference.sum()


class VelocityTrainer:
    def __init__(self, wm, policy, critic, lib, cfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.encoders['velocity'].parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        self.step_fn = VelocityStep(wm, lib.default_joint_pos, cfg.compile)

    def set_lr(self, lr):
        for group in self.opt.param_groups:
            group['lr'] = lr
        for group in self.critic_opt.param_groups:
            group['lr'] = lr * self.cfg.critic_lr/self.cfg.lr

    def update(self, buffer):
        cfg, wm, policy, lib = self.cfg, self.wm, self.policy, self.lib
        start = buffer.sample_starts(cfg.batch_size//2, cfg.history, slice(buffer.E//2, buffer.E))
        if start.command is None:
            raise ValueError('Mixed replay must store velocity command source and resampling countdown')
        xi = wm.spec.normalize_xi(start.xi)
        noise = OBS_NOISE_HALF.to(start.state.device)
        def observe(s,a):
            o = analytic_observation(s,a,start.xi,lib.default_joint_pos)
            return o + (2*torch.rand_like(o)-1)*noise if cfg.obs_noise else o
        hist = [observe(s,a) for s,a in zip(start.hist_states,start.hist_actions)]
        s, prev, frame = start.state, start.prev_action, start.frame
        meta = start.command.clone()
        survival = torch.ones(len(s), device=s.device)
        rewards, dones, features, commands, metas, survivals = [], [], [], [], [], [survival]
        align_sum = s.new_zeros(())
        align_count = s.new_zeros(())
        self.opt.zero_grad(set_to_none=True)
        # Decoder/world-model derivatives w.r.t. inputs remain available.
        with frozen(wm, policy.encoders['motion'], policy.decoder, self.critic):
            for t in range(cfg.horizon+1):
                command = command_at(meta, lib.velocity_commands[frame])
                features.append(wm.spec.canonical(s))
                commands.append(command)
                metas.append(meta.clone())
                if t == cfg.horizon:
                    break
                obs = flatten_history_term_major(torch.stack(hist))
                z = policy.encoders['velocity'](obs, command)
                with torch.no_grad():
                    goal = lib.goal_window(frame,cfg.future_frames)
                    if cfg.anchor_obs:
                        ref = reference_at(lib,frame)
                        b = wm.heads.robot_bodies(s)
                        i = wm.heads.anchor_index
                        goal = torch.cat([goal,anchor_relative_obs(b['pos'][:,i],b['quat'][:,i],ref.body_pos_w[:,i],ref.body_quat_w[:,i])],-1)
                    target = policy.encoders['motion'](goal)
                asum, count = masked_alignment(z, target, meta[:,3])
                align_sum, align_count = align_sum+asum, align_count+count
                a = policy.decoder(z,obs)
                nxt, r, d, obs_next = self.step_fn(s,a,prev,start.xi,command)
                rewards.append(r)
                dones.append(d)
                survival = survival*(1-d)
                survivals.append(survival)
                if cfg.obs_noise:
                    obs_next = obs_next + (2*torch.rand_like(obs_next)-1)*noise
                hist = hist[1:] + [obs_next]
                s,prev = nxt,a
                frame = torch.minimum(frame+1,lib.last_frame_of(frame))
                meta = advance(meta)
            feat, cmd, metadata = torch.stack(features),torch.stack(commands),torch.stack(metas)
            r,d,c = torch.stack(rewards),torch.stack(dones),torch.stack(survivals)
            discount = cfg.gamma**torch.arange(cfg.horizon,device=s.device)
            J = (discount[:,None]*c[:-1]*r).sum(0)
            if cfg.use_critic:
                J = J + cfg.gamma**cfg.horizon*c[-1]*self.critic(feat[-1],cmd[-1],xi,metadata[-1])
            J = J.mean()
            alignment = align_sum/align_count.clamp_min(1)
            # Exact specified loss: -J_vel + 0.1*MSE (no horizon rescaling).
            loss = -J + .1*alignment
            loss.backward()
        encoder = policy.encoders['velocity']
        allreduce_grads(encoder)
        gn = torch.nn.utils.clip_grad_norm_(encoder.parameters(),cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(encoder.parameters(),cfg.grad_clip_value)
        self.opt.step()
        metrics = {'velocity/objective':float(J.detach()),'velocity/alignment':float(alignment.detach()),
                   'velocity/loss':float(loss.detach()),'velocity/reference_fraction':float(metadata[:-1,:,3].mean()),
                   'velocity/grad_norm':float(gn),'velocity/reward':float(r.detach().mean()),
                   'velocity/survival_H':float(c[-1].detach().mean())}
        if cfg.use_critic:
            with torch.no_grad():
                feat,cmd,metadata = feat.detach(),cmd.detach(),metadata.detach()
                xis = xi[None].expand(cfg.horizon+1,-1,-1)
                values = self.critic(feat,cmd,xis,metadata)
                targets = gae_targets(r.detach(),d.detach(),values,cfg.gamma,cfg.gae_lambda).flatten()
                inputs = [x[:-1].flatten(0,1) for x in (feat,cmd,xis,metadata)]
            for _ in range(cfg.critic_epochs):
                for idx in torch.tensor_split(torch.randperm(len(targets),device=s.device),cfg.critic_minibatches):
                    if not len(idx):
                        continue
                    prediction = self.critic(*(x[idx] for x in inputs))
                    closs = (prediction-targets[idx]).square().mean()
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(),cfg.critic_grad_clip)
                    self.critic_opt.step()
            metrics['velocity/critic_loss'] = float(closs.detach())
        return metrics
