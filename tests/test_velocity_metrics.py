import math
import torch
from supermimic.velocity.metrics import VelocityMetrics


def test_heading_source_masks_and_falls():
    stats=VelocityMetrics('cpu')
    # Four environments: motion (ignored), reference, random nonzero, zero.
    quat=torch.tensor([[1.,0,0,0],[math.sqrt(.5),0,0,math.sqrt(.5)],[1.,0,0,0],[1.,0,0,0]])
    lin=torch.tensor([[999.,999,0],[0.,2.,0],[2.,0,0],[.3,.4,0]])
    ang=torch.zeros(4,3);ang[2,2]=1
    cmd=torch.tensor([[0.,0,0],[1.,0,0],[1.,0,.5],[0.,0,0]])
    meta=torch.zeros(4,5);meta[1,3]=1
    mask=torch.tensor([False,True,True,True]);falls=torch.tensor([True,False,True,False])
    stats.update(quat,lin,ang,cmd,meta,mask,falls)
    # Mutating next-step command/source cannot change already accumulated statistics.
    cmd.zero_();meta.zero_()
    result=stats.flush()
    assert result['all/samples']==3
    assert abs(result['reference/vx_mae']-1)<1e-6
    assert abs(result['reference/vy_mae'])<1e-6
    assert abs(result['random/wz_mae']-.5)<1e-6
    assert abs(result['zero/xy_error']-.5)<1e-6
    assert abs(result['all/falls_per_1k_steps']-1000/3)<1e-5
    assert result['random/falls_per_1k_steps']==1000
    empty=stats.flush()
    assert empty['all/samples']==0 and math.isnan(empty['all/vx_mae'])
