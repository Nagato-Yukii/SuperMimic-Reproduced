# LAFAN heading velocity augmentation

`uv run python scripts/data/add_heading_velocity.py` creates
`data/npz/lafan_w_velocity` from `data/npz/lafan1`, preserving every original
NPZ field exactly. Existing output directories are refused.

New fields: `vx`, `vy`, `wz`, each `(T,)`; `velocity_command`, `(T,3)`, in that
order. Units: m/s, m/s, rad/s. Existing tracking observations are unchanged.

At 50 Hz, the default adjacent central difference uses
`lo=max(t-1,0)`, `hi=min(t+1,T-1)`, `dt=(hi-lo)/50`.
Thus interior frames divide by 0.04 s; the first and last divide by 0.02 s.
A single-frame clip receives zero velocity. No 25-frame smoothing is applied.

The root is `pelvis`. World linear velocity `(p[hi]-p[lo])/dt` is rotated into
the centre frame's yaw-only heading axes: x forward, y left, z up.
World angular velocity is `Log(R[hi] R[lo]^T)/dt`, using the shortest SO(3)
rotation. A yaw-only coordinate change leaves z unchanged, giving `wz`.
This is vertical angular velocity, not Euler yaw differentiation for tilted
poses. Quaternions in the source NPZ are wxyz.

`velocity_metadata.json` records the exact convention and clip frame counts.
Conversion checks every saved original field for equality and all velocity
values for finiteness. Tests: `uv run pytest tests/test_heading_velocity.py -q`.

The superseded 50-frame-span output is retained separately as
`data/npz/lafan_w_velocity_previous_50frame_span`; it is not the corrected data.
