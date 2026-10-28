"""Record a continuous velocity-command rollout with measured heading velocities."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json
import subprocess
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.kinematics.rotations import quat_apply_inverse, yaw_quat

@torch.no_grad()
def main():
    out=Path('logs/motion_velocity_lafan1/videos');out.mkdir(parents=True,exist_ok=True)
    ckpt=Path('logs/motion_velocity_lafan1/ckpt_0093000.pt')
    policy,cfg=load_policy_from_checkpoint(ckpt,'cuda:0')
    files=discover_npz(Path('data/npz/lafan_w_velocity'),names=['walk1_subject1'])
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=2,
        future_frames=cfg['env']['future_frames'],episode_length_s=1e5,action_noise_std=0,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,seed=0,anchor_obs=cfg['env'].get('anchor_obs',False),velocity_enabled=True),device='cuda:0')
    cmd=env.command_manager.get_term('motion');cmd.velocity_mask[:]=True
    cmd.forced_frame[:]=cmd.library.clip_start[0]+100
    obs,_=env.reset()
    model=env.sim.mj_model
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=720,width=1280)
    camera=mujoco.MjvCamera();camera.distance=3.7;camera.azimuth=135;camera.elevation=-20
    option=mujoco.MjvOption();option.geomgroup[0]=1;option.geomgroup[1]=1;option.geomgroup[2]=1
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    segments=[('Forward',(1.,0.,0.)),('Left sidestep',(0.,.5,0.)),('Turn left',(0.,0.,1.)),
              ('Forward + turn',(1.,0.,.7)),('Backward',(-.5,0.,0.)),('Stop',(0.,0.,0.))]
    video=out/'velocity_commands_0093000.mp4'
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720',
        '-r','25','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    rows=[];falls=0
    try:
        for segment,(label,command) in enumerate(segments):
            print('Recording',label,flush=True)
            for t in range(250):
                cmd.velocity_metadata[:,3]=0;cmd.velocity_metadata[:,4]=1e8
                cmd.velocity_metadata[:,:3]=torch.tensor(command,device='cuda:0')
                action=policy(obs['policy'],cmd.velocity_command,interface='velocity')
                obs,_,term,tout,_=env.step(action)
                falls+=int(term[0].item())
                robot=cmd.robot.data;ri=cmd.robot_root_body_index
                v=quat_apply_inverse(yaw_quat(robot.body_link_quat_w[:,ri]),robot.body_link_lin_vel_w[:,ri])
                actual=[float(v[0,0]),float(v[0,1]),float(robot.body_link_ang_vel_w[0,ri,2])]
                elapsed=(segment*250+t+1)*.02
                rows.append([elapsed,*command,*actual,int(term[0]),int(tout[0])])
                if t%2:continue
                data.qpos[:]=env.sim.data.qpos[:][0].cpu().numpy()
                data.qvel[:]=env.sim.data.qvel[:][0].cpu().numpy()
                mujoco.mj_forward(model,data)
                camera.lookat[:]=data.qpos[:3];camera.lookat[2]=.75
                renderer.update_scene(data,camera=camera,scene_option=option)
                # Add a world-fixed floor grid to make translation visible.
                scene=renderer.scene
                for axis in range(2):
                    for k in range(-30,31):
                        g=scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.35,.42,.48,1.]))
                        a=np.array([-30.,float(k),.006]) if axis==0 else np.array([float(k),-30.,.006])
                        b=np.array([30.,float(k),.006]) if axis==0 else np.array([float(k),30.,.006])
                        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.,a,b)
                        scene.ngeom+=1
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,1280,133),fill=(17,24,39))
                draw.text((26,12),'SuperMimic | Velocity commands | checkpoint 93,000',font=titlefont,fill='white')
                draw.text((26,55),f'{label}   |   {elapsed:05.2f} / 30.00 s   |   continuous rollout, 1x speed',font=font,fill=(190,205,225))
                draw.text((26,94),'Heading frame: x forward, y left | vx, vy: m/s | wz: rad/s',font=font,fill=(190,205,225))
                draw.rectangle((0,610,1280,720),fill=(17,24,39))
                draw.text((26,624),f'TARGET    vx {command[0]:+5.2f}     vy {command[1]:+5.2f}     wz {command[2]:+5.2f}',font=titlefont,fill=(255,201,82))
                draw.text((26,667),f'ACTUAL    vx {actual[0]:+5.2f}     vy {actual[1]:+5.2f}     wz {actual[2]:+5.2f}',font=titlefont,fill=(91,224,220))
                draw.text((1040,667),f'Falls: {falls}',font=font,fill='white')
                encoder.stdin.write(np.asarray(frame).tobytes())
                if segment==0 and t==100:frame.save(out/'preview.png')
        np.savetxt(out/'velocity_commands_0093000.csv',np.array(rows),delimiter=',',header='time,target_vx,target_vy,target_wz,actual_vx,actual_vy,actual_wz,terminated,timeout',comments='')
        (out/'velocity_commands_0093000.json').write_text(json.dumps({'checkpoint':str(ckpt),'seconds':30,'falls':falls,'segments':segments},indent=2))
    finally:
        encoder.stdin.close();code=encoder.wait();renderer.close();env.close()
    if code:raise RuntimeError(f'ffmpeg failed {code}')
    print('Saved',video,'falls',falls,flush=True)

if __name__=='__main__':main()
