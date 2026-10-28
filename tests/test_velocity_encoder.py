import pytest
import torch
from torch import nn
from supermimic.policy.nets import Policy, PolicyCfg, MotionEncoder, VelocityEncoder


def test_default_architecture_and_latents():
    cfg = PolicyCfg()
    motion, velocity = MotionEncoder(cfg), VelocityEncoder(cfg)
    for encoder, input_dim in [(motion, 680), (velocity, 933)]:
        layers = [m for m in encoder.mlp.net if isinstance(m, nn.Linear)]
        assert [m.in_features for m in layers] == [input_dim, 2048, 1024, 512, 256]
        assert [m.out_features for m in layers] == [2048, 1024, 512, 256, 32]
    assert motion(torch.zeros(2, 680)).shape == (2, 32)
    z = velocity(torch.randn(2, 930), torch.randn(2, 3))
    assert z.shape == (2, 32)
    assert (z.abs() <= 1).all()


def test_shared_decoder_and_velocity_gradients_with_frozen_decoder():
    policy = Policy(PolicyCfg(encoder_hidden=(32, 16), decoder_hidden=(32, 16), velocity_encoder=True))
    policy.decoder.requires_grad_(False)
    obs, cmd = torch.randn(3, 930), torch.randn(3, 3)
    action = policy(obs, cmd, interface='velocity')
    torch.testing.assert_close(action, policy.decoder(policy.encoders['velocity'](obs, cmd), obs))
    assert action.shape == (3, 29)
    action.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoders['velocity'].parameters())
    assert all(p.grad is None for p in policy.decoder.parameters())
    assert all(p.grad is None for p in policy.encoders['motion'].parameters())
    assert policy(obs, torch.randn(3,680)).shape == (3,29)
    copy = Policy(policy.cfg)
    copy.load_state_dict(policy.state_dict())
    torch.testing.assert_close(action, copy(obs,cmd,interface='velocity'))


def test_motion_default_remains_legacy_compatible():
    policy = Policy(PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,)))
    assert set(policy.encoders) == {'motion'}
    clone = Policy(policy.cfg)
    clone.load_state_dict(policy.state_dict(), strict=True)
    with pytest.raises(ValueError, match='Enable'):
        policy(torch.zeros(1,930),torch.zeros(1,3),interface='velocity')
