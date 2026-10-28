"""Simulator-only command tracking statistics, accumulated before reset/resampling."""
import torch
from supermimic.kinematics.rotations import quat_apply_inverse, yaw_quat


class VelocityMetrics:
    groups = ('all', 'reference', 'random', 'zero')
    columns = ('samples', 'vx_mae', 'vy_mae', 'wz_mae', 'xy_error',
               'linear_score', 'yaw_score', 'falls')

    def __init__(self, device):
        self.sums = torch.zeros(4, len(self.columns), device=device)

    @torch.no_grad()
    def update(self, quat, lin_w, ang_w, command, metadata, velocity_mask, falls):
        measured = quat_apply_inverse(yaw_quat(quat), lin_w)
        error = torch.cat([measured[:, :2]-command[:, :2], ang_w[:, 2:3]-command[:, 2:3]], -1)
        xy_sq = error[:, :2].square().sum(-1)
        reference = metadata[:, 3].bool()
        zero = ~reference & (command == 0).all(-1)
        random = ~reference & ~zero
        masks = torch.stack([velocity_mask, velocity_mask & reference,
                             velocity_mask & random, velocity_mask & zero]).float()
        values = torch.cat([torch.ones_like(error[:, :1]), error.abs(), xy_sq.sqrt()[:, None],
                            torch.exp(-xy_sq/.25)[:, None], torch.exp(-error[:, 2].square()/.5)[:, None],
                            falls.float()[:, None]], -1)
        self.sums += masks @ values

    @torch.no_grad()
    def flush(self):
        sums = self.sums.clone()
        self.sums.zero_()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
        rows = sums.cpu().tolist()  # one host transfer per log interval
        out = {}
        for group,row in zip(self.groups,rows):
            count = row[0]
            out[f'{group}/samples'] = count
            for name,value in zip(self.columns[1:],row[1:]):
                if name == 'falls':
                    out[f'{group}/falls_per_1k_steps'] = 1000*value/count if count else float('nan')
                else:
                    out[f'{group}/{name}'] = value/count if count else float('nan')
        return out
