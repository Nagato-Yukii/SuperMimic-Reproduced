# Mixed motion / RL velocity interface

Enable `policy.velocity_encoder` to use the mixed training path. The preset
`configs/motion_velocity_5070ti.yaml` initializes the shared world model, motion
encoder, decoder and motion critic from the local LAFAN tracking checkpoint.
Initialization starts a **new** schedule and optimizers. `resume` instead
restores a previously saved mixed run, including both critics and optimizers.

```bash
uv run sm-train --config configs/motion_velocity_5070ti.yaml
```

This command launches training; implementation tests do not start a full run.
Both encoders retain hidden widths `[2048,1024,512,256]`, SiLU activations and
32-dimensional tanh latents. Velocity input is 930 history features plus 3 SI
heading command features. Both interfaces use the same decoder.

## Sampling

Half the simulator environments use motion tracking, half velocity: 2048 each
with the preset. Each actor update samples half its total imagined batch from
each corresponding replay partition (256 each for batch size 512). Both train
one shared dynamics model from replay containing both interfaces.

On reset and every 250 control steps, velocity command source is sampled with
probability 0.5 reference / 0.5 random. Reference commands are read every step
from `velocity_command` in the motion NPZ at the corresponding frame. Plain
NPZs without that field use the same adjacent-frame heading-velocity computation.
Packed `.pt` motion libraries are currently rejected for mixed training.

Random commands use independent uniforms `vx ∈ [-1.5,2]`, `vy ∈ [-1.32,1.32]`,
`wz ∈ [-3.35,3.35]`; with probability 0.1 the complete command is zeroed. Random
commands stay fixed until reset/resampling. The overall expected fractions are
50% motion, 25% velocity/reference, 22.5% velocity/random, 2.5% velocity/zero.
Replay records the held command, source flag and remaining resampling steps;
imagination continues that schedule rather than re-drawing at rollout start.

## Objectives and parameter ownership

Both velocity command sources use the **same velocity reward**, never full-body
tracking reward. Sum the following rates and multiply once by dt=0.02:

- `2 exp(-||v_xy-command_xy||² / 0.5²)` in pelvis heading coordinates.
- `2 exp(-(omega_world_z-command_z)² / 0.5)`.
- `-0.3 ||action-previous_action||²`.
- `-2 sum(max(q-qmax,0)+max(qmin-q,0))` using soft joint limits.
- `-4 sin(roll_torso)²` (permits forward pitch).
- `-0.05 (omega_torso_body_x² + omega_torso_body_y²)`.

Velocity fall termination: pelvis height <0.35 m or pelvis tilt >73 degrees.
Episode timeout and reference clip-end remain truncations, not falls. Motion
retains its original reference-relative height/tilt termination and rewards.
One shared world model has separate learned termination heads for the two rules,
trained on the same simulator state transitions using their respective labels.

Velocity actor loss is exactly `-J_velocity + 0.1 L_alignment`, with discounted
survival-weighted rollout return and terminal velocity critic bootstrap.
Alignment is mean squared error over 32 latent components, then averaged over
**reference-source samples only** across the rollout. Its target is the
same-time motion latent with stop-gradient. Random/zero-source samples have no
alignment loss. The decoder, motion encoder, world model and velocity critic
parameters are frozen during this actor update, with input derivatives retained.
Only the velocity encoder optimizer steps. Motion BPTT updates only the motion
encoder and decoder. Each interface has a separate critic and GAE regression.

Velocity BPTT compiles each differentiable world-model/reward/termination/observation
step using `policy_trainer.compile`, like the motion branch. The preset inherits
`default`; the first update may spend several minutes compiling on the CPU.
Setting the option to null preserves the eager fallback. Logs separately report
velocity update time (included in total policy time). Benchmark the full-size
256-sample, 32-step velocity update with:

```bash
uv run python scripts/bench_velocity_update.py --modes eager default
```

The benchmark synchronizes CUDA, excludes one cold call from the repeated timing,
and checks compiled forward values and state/action gradients against eager.
It uses reference-state fixtures, not a complete simulator training iteration.

Tests cover reward denominators, command distributions and 250-step hold,
reference-only stop-gradient alignment, falls, a CUDA mixed training smoke run,
parameter isolation and checkpoint restore. The 93,000-iteration checkpoint has also been evaluated; see [evaluation results](evaluation_0093000.md).

## Observing command following

Each console log interval now includes `vel_sim` diagnostics measured in the
simulator reward hook **before** automatic reset, reference advance or command
resampling. Measurements include terminal steps and exclude motion-interface
environments. The CUDA accumulator transfers statistics to the CPU only when
logging. Distributed runs sum counts and errors across ranks before averaging.

- MAE vx/vy/wz: component absolute error (m/s, m/s, rad/s), lower is better.
- xy_err: mean Euclidean planar velocity error, separately for reference,
  random-nonzero and explicitly zero commands. wz_err uses the same grouping.
- track_score xy/wz: exponential tracking scores in [0,1], higher is better;
  excludes posture/action penalties and is not an episode success rate.
- falls/1k_steps: fall terminations per 1000 velocity-environment control steps;
  excludes timeouts/clip ends. This is a failure frequency, not a success rate.
- vel_imag_reward and align: world-model reward and reference latent alignment.

TensorBoard tags: `velocity_sim/{all,reference,random,zero}/*`, plus existing
`policy/velocity/*` (now also `survival_H`). Missing groups report nan in the
console and are not logged as zero-valued TensorBoard errors.

`train_ret` still mixes both interfaces; the original `imag_ret/surv` belongs
to motion. Compare simulator error trends by command source, since varying
commands and episode lengths affect aggregate training statistics. These online
metrics are not a fixed-command held-out evaluation.

Code changes do not hot-patch an already running Python process. Resume a saved
mixed checkpoint to pick up the new logging. The preset initializes from the
motion checkpoint by default, so clear that option when resuming:

```bash
uv run sm-train --config configs/motion_velocity_5070ti.yaml \
  --initialize-from None --resume logs/motion_velocity_lafan1/ckpt_latest.pt
```

## Evaluation and inference video

Run `uv run python scripts/eval_velocity.py --ckpt PATH --out results.json` for fixed and reference velocity commands. Local motion data and checkpoints are required and are not included in this repository.

`uv run python scripts/record_velocity.py` records a 30-second continuous command sequence using `logs/motion_velocity_lafan1/ckpt_0093000.pt`. It requires FFmpeg, Pillow, EGL rendering, and the DejaVu Sans font. Output is written under `logs/motion_velocity_lafan1/videos/`, with target/measured heading velocities and fall counts overlaid.
