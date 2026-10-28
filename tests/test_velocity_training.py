import math
from pathlib import Path
import pytest
import torch
from supermimic.velocity.task import resample, advance, command_at, reward_terms, terminated
from supermimic.velocity.trainer import masked_alignment


def test_command_sampling_hold_and_resample():
    torch.manual_seed(123)
    meta = torch.zeros(20000,5)
    resample(meta,torch.arange(len(meta)))
    assert abs(float(meta[:,3].mean())-.5)<.02
    random = meta[:,3]==0
    assert abs(float((meta[random,:3]==0).all(-1).float().mean())-.1)<.02
    assert ((meta[:,:3]>=torch.tensor([-1.5,-1.32,-3.35])) & (meta[:,:3]<=torch.tensor([2,1.32,3.35]))).all()
    held = meta.clone()
    for _ in range(249):
        meta = advance(meta)
    torch.testing.assert_close(meta[:,:4],held[:,:4])
    assert (meta[:,4]==1).all()
    meta = advance(meta)
    assert (meta[:,4]==250).all()
    assert not torch.equal(meta[:,:3],held[:,:3])
    reference = torch.randn(len(meta),3)
    cmd = command_at(meta,reference)
    torch.testing.assert_close(cmd[meta[:,3].bool()],reference[meta[:,3].bool()])
    torch.testing.assert_close(cmd[~meta[:,3].bool()],meta[~meta[:,3].bool(),:3])


def test_reference_only_alignment_stopgrad():
    z = torch.ones(3,32,requires_grad=True)
    target = torch.zeros_like(z,requires_grad=True)
    total,count = masked_alignment(z,target,torch.tensor([1.,0.,1.]))
    loss=.1*total/count
    loss.backward()
    assert target.grad is None and z.grad[1].abs().sum()==0
    torch.testing.assert_close(loss,torch.tensor(.1))
    zero,_=masked_alignment(z,target,torch.zeros(3))
    assert zero==0


def test_velocity_reward_and_fall_rule():
    q=torch.tensor([[1.,0,0,0]])
    zero3=torch.zeros(1,3);j=torch.zeros(1,29)
    limits=torch.tensor([-1.,1.]).expand(29,2)
    terms=reward_terms(q,zero3,zero3,q,zero3,j,limits,j,j,zero3)
    torch.testing.assert_close(sum(terms.values())*.02,torch.tensor([.08]))
    cmd=torch.tensor([[.5,0,math.sqrt(.5)]])
    terms=reward_terms(q,zero3,zero3,q,zero3,j,limits,j,j,cmd)
    torch.testing.assert_close(terms['linear'],torch.tensor([2/math.e]))
    torch.testing.assert_close(terms['yaw'],torch.tensor([2/math.e]))
    s=torch.zeros(3,71);s[:,3]=1;s[:,2]=torch.tensor([.34,.35,1.])
    s[2,3]=math.cos(math.radians(74)/2);s[2,4]=math.sin(math.radians(74)/2)
    assert terminated(s).tolist()==[True,False,True]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_mixed_training_smoke(tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    cfg=TrainConfig(run_name='mixed',log_root=tmp_path)
    cfg.policy.velocity_encoder=True
    cfg.data.npz_dir=Path('data/npz/lafan_w_velocity')
    cfg.data.clips=['walk1_subject1']
    cfg.env.num_envs=8;cfg.env.replay_capacity=64
    cfg.env.obs_noise=False;cfg.env.domain_randomization=False;cfg.env.reset_noise=False
    cfg.world_model.hidden=(32,32);cfg.world_model.term_hidden=(16,)
    cfg.policy.encoder_hidden=(32,16);cfg.policy.decoder_hidden=(32,16)
    cfg.critic.hidden=(16,)
    cfg.wm_trainer.compile=None;cfg.policy_trainer.compile=None
    cfg.wm_trainer.window=2;cfg.wm_trainer.batches_per_update=1;cfg.wm_trainer.batch_size=8
    cfg.policy_trainer.batch_size=8;cfg.policy_trainer.horizon=2
    cfg.policy_trainer.critic_epochs=1;cfg.policy_trainer.critic_minibatches=1
    cfg.schedule.total_iterations=12;cfg.schedule.warmup_iterations=5
    cfg.schedule.log_interval=12;cfg.schedule.checkpoint_interval=12
    tr=Trainer(cfg)
    try:
        assert tr.command.velocity_mask.tolist()==[False]*4+[True]*4
        tr.run()
        assert tr.num_updates>0
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(tmp_path/'mixed/tb'));events.Reload()
        assert events.Scalars('velocity_sim/all/samples')[-1].value == 4*12
        assert events.Scalars('velocity_sim/all/vx_mae')[-1].value >= 0
        # Velocity update changes only velocity encoder (plus its own critic).
        motion={k:v.clone() for k,v in tr.policy.encoders['motion'].state_dict().items()}
        decoder={k:v.clone() for k,v in tr.policy.decoder.state_dict().items()}
        wm={k:v.clone() for k,v in tr.wm.state_dict().items()}
        velocity={k:v.clone() for k,v in tr.policy.encoders['velocity'].state_dict().items()}
        original_reward = tr.wm.heads.reward
        def reject_tracking(*args, **kwargs):
            raise AssertionError("Velocity rollout must never use tracking rewards")
        tr.wm.heads.reward = reject_tracking
        tr.buffer.command[..., 3] = 0
        tr.buffer.command[..., 4] = 250
        random_metrics = tr.velocity_trainer.update(tr.buffer)
        assert random_metrics['velocity/alignment'] == 0
        tr.buffer.command[..., 3] = 1
        metrics=tr.velocity_trainer.update(tr.buffer)
        assert metrics['velocity/reference_fraction'] == 1
        assert metrics['velocity/alignment'] > 0
        tr.wm.heads.reward = original_reward
        assert all(math.isfinite(v) for v in metrics.values())
        for module,old in [(tr.policy.encoders['motion'],motion),(tr.policy.decoder,decoder),(tr.wm,wm)]:
            for k,v in module.state_dict().items():
                torch.testing.assert_close(v,old[k],rtol=0,atol=0)
        assert any(not torch.equal(v,velocity[k]) for k,v in tr.policy.encoders['velocity'].state_dict().items())
        # Exact resume restores each branch optimizer and critic.
        tr.load(tmp_path/'mixed/ckpt_latest.pt')
        assert tr.iteration==12
        assert tr.velocity_trainer.opt.state
    finally:
        tr.env.close()
