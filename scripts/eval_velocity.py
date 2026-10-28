"""Deterministic velocity evaluation: held commands and full reference-velocity clips."""
from pathlib import Path
import json
import torch
import tyro
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.kinematics.rotations import quat_apply_inverse, yaw_quat

CASES = [('stop',(0,0,0)),('forward_0.5',(.5,0,0)),('forward_1',(1,0,0)),('forward_2',(2,0,0)),
         ('backward_0.5',(-.5,0,0)),('backward_1.5',(-1.5,0,0)),
         ('left_0.5',(0,.5,0)),('right_0.5',(0,-.5,0)),('left_1.32',(0,1.32,0)),('right_1.32',(0,-1.32,0)),
         ('yaw_left_1',(0,0,1)),('yaw_right_1',(0,0,-1)),('yaw_left_3.35',(0,0,3.35)),('yaw_right_3.35',(0,0,-3.35)),
         ('combined',(1,.5,1)),('combined_reverse',(-1,-.5,-1))]
COLS = ('vx','vy','wz','vx_mae','vy_mae','wz_mae','xy_error','linear_score','yaw_score')

class Meter:
    def __init__(self,n,device,warmup):
        self.active=torch.ones(n,device=device,dtype=torch.bool)
        self.count=torch.zeros(n,device=device)
        self.steady_count=torch.zeros_like(self.count)
        self.sums=torch.zeros(n,len(COLS),device=device)
        self.steady=torch.zeros_like(self.sums)
        self.warmup=warmup

    def update(self,quat,lin_w,ang_w,command,metadata,velocity_mask,falls):
        # Invoked by sim_reward before automatic reset and command advance.
        v=quat_apply_inverse(yaw_quat(quat),lin_w)
        actual=torch.cat([v[:,:2],ang_w[:,2:3]],-1)
        err=actual-command
        xy=err[:,:2].square().sum(-1)
        values=torch.cat([actual,err.abs(),xy.sqrt()[:,None],torch.exp(-xy/.25)[:,None],torch.exp(-err[:,2:3].square()/.5)],-1)
        valid=self.active & velocity_mask
        steady=valid & (self.count>=self.warmup)
        self.sums+=values*valid[:,None]
        self.steady+=values*steady[:,None]
        self.count+=valid
        self.steady_count+=steady

    def summary(self,ids,failed,complete):
        n=len(ids)
        def means(sums,counts):
            c=float(counts[ids].sum())
            return dict(zip(COLS,(sums[ids].sum(0)/c).cpu().tolist())) if c else None
        return {'episodes':n,'failed':int(failed[ids].sum()),'completed':int(complete[ids].sum()),
                'completion_rate':float(complete[ids].float().mean()),'mean_seconds':float(self.count[ids].mean())*.02,
                'all_steps':means(self.sums,self.count),'after_2s':means(self.steady,self.steady_count)}

@torch.no_grad()
def evaluate(policy,cfg,files,suite,repeats,seconds,device):
    n=len(files) if suite=='reference' else len(CASES)*repeats
    args=SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=n,future_frames=cfg['env']['future_frames'],
        episode_length_s=1e5,action_noise_std=0,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,seed=0,anchor_obs=cfg['env'].get('anchor_obs',False),velocity_enabled=True)
    env=make_env(args,device=device)
    try:
        cmd=env.command_manager.get_term('motion');lib=cmd.library
        cmd.velocity_mask[:]=True
        if suite=='reference':
            cmd.forced_frame[:]=lib.clip_start
        else:
            # Same eight walking phase starts for every case; no random reset perturbation.
            cmd.forced_frame[:]=lib.clip_start[0]+100+50*(torch.arange(n,device=device)%repeats)
        obs,_=env.reset()
        cmd.velocity_metadata[:,4]=1e8  # hold source/command throughout the evaluation
        if suite=='reference':
            cmd.velocity_metadata[:,3]=1
            max_steps=max(c.length for c in lib.clips)+1
        else:
            cmd.velocity_metadata[:,3]=0
            cmd.velocity_metadata[:,:3]=torch.tensor([c for _,c in CASES],device=device).repeat_interleave(repeats,dim=0)
            max_steps=round(seconds/.02)
        meter=Meter(n,device,100);cmd.velocity_stats=meter
        failed=torch.zeros(n,device=device,dtype=torch.bool);complete=torch.zeros_like(failed)
        for t in range(max_steps):
            a=policy(obs['policy'],cmd.velocity_command,interface='velocity')
            obs,rew,term,tout,_=env.step(a)
            failed |= meter.active & term
            if suite=='reference':complete |= meter.active & tout & ~term
            meter.active &= ~(term|tout)
            if not meter.active.any():break
        if suite=='fixed':complete |= meter.active
        groups={}
        if suite=='fixed':
            for i,(name,command) in enumerate(CASES):
                groups[name]={'command':command,**meter.summary(torch.arange(i*repeats,(i+1)*repeats,device=device),failed,complete)}
        else:
            for i,clip in enumerate(lib.clips):
                groups[clip.name]=meter.summary(torch.tensor([i],device=device),failed,complete)
        return {'summary':meter.summary(torch.arange(n,device=device),failed,complete),'cases':groups}
    finally:env.close()


def main(ckpt:Path, out:Path, suites:tuple[str,...]=('fixed','reference'), repeats:int=8,seconds:float=20.,device:str='cuda:0'):
    policy,cfg=load_policy_from_checkpoint(ckpt,device)
    assert cfg['policy']['velocity_encoder']
    results={'checkpoint':str(ckpt),'protocol':{'noise':False,'domain_randomization':False,'fixed_seconds':seconds,'repeats':repeats,
        'warmup_excluded_seconds':2,'fixed_starts':'walk1_subject1 frames 100 + 50*i (i=0..repeats-1)',
        'termination':'height < 0.35m or tilt > 73 degrees; first failure stops scoring',
        'error_denominator':'executed steps before first failure; not a fixed-horizon failure penalty'}}
    for suite in suites:
        if suite not in ('fixed','reference'):raise ValueError(suite)
        files=discover_npz(Path('data/npz/lafan_w_velocity'),names=['walk1_subject1'] if suite=='fixed' else None)
        results[suite]=evaluate(policy,cfg,files,suite,repeats,seconds,device)
        print(suite,json.dumps(results[suite]['summary']),flush=True)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2)+'\n')
    print('saved',out,flush=True)

if __name__=='__main__':tyro.cli(main)
