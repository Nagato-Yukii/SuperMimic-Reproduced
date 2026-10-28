"""Bounded synchronized velocity-update benchmark (no simulator or checkpoint writes)."""
import cProfile
import json
import time
from pathlib import Path
import torch
import tyro
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.velocity.trainer import VelocityTrainer, VelocityCritic


def main(modes: tuple[str,...] = ('eager',), batch: int = 256, horizon: int = 32, reps: int = 2,
         out: Path = Path('logs/hardware_checks/velocity_baseline.json')):
    torch.manual_seed(7)
    dev='cuda:0'
    model=load_g1_model(); q0=torch.zeros(29,device=dev)
    lib=MotionLibrary(discover_npz(Path('data/npz/lafan_w_velocity'),limit=1),TRACKED_BODIES,q0,device=dev)
    heads=AnalyticHeads(G1Kinematics(model,device=dev),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
                        torch.as_tensor(model.jnt_range[1:],device=dev,dtype=torch.float32),q0)
    wm=WorldModel(WorldModelCfg(velocity_enabled=True),heads,dev)
    policy=Policy(PolicyCfg(velocity_encoder=True)).to(dev)
    ck=torch.load('logs/motion_lafan1/ckpt_latest.pt',map_location=dev,weights_only=False)
    wm.load_state_dict(ck['wm'],strict=False);policy.load_state_dict(ck['policy'],strict=False)
    del ck
    frame=torch.arange(batch,device=dev)+100
    qp,qv=lib.ref_qpos_qvel(frame);s=torch.cat([qp,qv],-1)
    meta=torch.zeros(batch,5,device=dev);meta[:batch//2,3]=1;meta[:,4]=250
    start=StartBatch(s,torch.tensor([.75]+[0.]*32,device=dev).expand(batch,-1).clone(),frame,
                     torch.zeros(batch,29,device=dev),s[None].expand(10,-1,-1).contiguous(),
                     torch.zeros(10,batch,29,device=dev),meta)
    class Buffer:
        E=2*batch
        def sample_starts(self,*args):return start
    results=[]
    for mode in modes:
        critic=VelocityCritic().to(dev)
        tr=VelocityTrainer(wm,policy,critic,lib,PolicyTrainerCfg(batch_size=batch*2,horizon=horizon,compile=None if mode=='eager' else mode))
        times=[];torch.cuda.reset_peak_memory_stats()
        for i in range(reps+1):
            torch.cuda.synchronize();t=time.perf_counter()
            prof=cProfile.Profile() if i==0 and mode=='eager' else None
            if prof:prof.enable()
            metrics=tr.update(Buffer())
            if prof:prof.disable()
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            print(f'{mode} update {i}: {elapsed:.3f}s',flush=True)
            if i==0:
                cold=elapsed
                if prof:
                    import pstats
                    pstats.Stats(prof).sort_stats('cumtime').print_stats(18)
            else:times.append(elapsed)
        # Check the actual compiled forward and state/action derivatives against eager.
        from supermimic.velocity.trainer import frozen
        with frozen(wm):
            state = start.state.detach().clone().requires_grad_(True)
            action = torch.zeros(batch,29,device=dev,requires_grad=True)
            args = (state,action,start.prev_action,start.xi,lib.velocity_commands[start.frame])
            eager = tr.step_fn._step(*args)
            candidate = tr.step_fn(*args)
            for expected, actual in zip(eager,candidate):
                torch.testing.assert_close(actual,expected,rtol=2e-3,atol=3e-5)
            grad_e = torch.autograd.grad(sum(x.square().mean() for x in eager),(state,action))
            grad_c = torch.autograd.grad(sum(x.square().mean() for x in candidate),(state,action))
            for expected,actual in zip(grad_e,grad_c):
                torch.testing.assert_close(actual,expected,rtol=2e-3,atol=3e-5)
            print(f'{mode}: forward and state/action gradient equivalence passed',flush=True)
        results.append(dict(mode=mode,cold_s=cold,times_s=times,mean_s=sum(times)/len(times),peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9,metrics=metrics))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,batch=batch,horizon=horizon,dtype='float32',results=results),indent=2))
    print(out,flush=True)
if __name__=='__main__':tyro.cli(main)
