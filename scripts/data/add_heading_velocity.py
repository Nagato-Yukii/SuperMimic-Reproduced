"""Preserve complete motion NPZs and add heading-frame velocity commands."""
from pathlib import Path
import json
import numpy as np
import tyro
from supermimic.data.heading_velocity import heading_velocity


def main(input_dir: Path = Path('data/npz/lafan1'), output_dir: Path = Path('data/npz/lafan_w_velocity'), half_window: int = 1):
    files = sorted(input_dir.glob('*.npz'))
    if not files:
        raise ValueError(f'No motion files in {input_dir}')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    output_dir.mkdir(parents=True)
    records = []
    for path in files:
        with np.load(path, allow_pickle=False) as src:
            data = {k: src[k] for k in src.files}
        if {'vx', 'vy', 'wz', 'velocity_command'} & data.keys():
            raise ValueError(f'{path}: velocity fields already exist')
        root = list(data['body_names']).index('pelvis')
        fps = float(data['fps'].reshape(-1)[0])
        velocity = heading_velocity(data['body_pos_w'][:, root], data['body_quat_w'][:, root], fps, half_window)
        target = output_dir / path.name
        np.savez_compressed(target, **data, vx=velocity[:, 0], vy=velocity[:, 1], wz=velocity[:, 2], velocity_command=velocity)
        with np.load(target, allow_pickle=False) as out:
            for key, value in data.items():
                np.testing.assert_array_equal(out[key], value)
            assert out['velocity_command'].shape == (len(data['joint_pos']), 3)
            assert np.isfinite(out['velocity_command']).all()
        records.append({'file':path.name, 'frames':len(velocity), 'fps':fps})
        print(f'{len(records)}/{len(files)} {path.name}: {len(velocity)} frames', flush=True)
    metadata = {'source':str(input_dir.resolve()), 'root_body':'pelvis', 'coordinate_frame':'centre-frame pelvis heading (yaw only)',
                'columns':['vx','vy','wz'], 'units':['m/s','m/s','rad/s'], 'half_window_frames':half_window,
                'difference':'lo=max(t-h,0); hi=min(t+h,T-1); difference divided by (hi-lo)/fps; single-frame velocity zero',
                'angular_difference':'Log(R[hi] R[lo]^T)/((hi-lo)/fps), shortest SO(3) rotation, world z equals heading z; not Euler yaw derivative',
                'fields':'vx,vy,wz: (T,); velocity_command: (T,3); all original fields unchanged',
                'clips':records, 'total_frames':sum(r['frames'] for r in records)}
    (output_dir/'velocity_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Verified {len(records)} clips, {metadata["total_frames"]} frames -> {output_dir}')

if __name__ == '__main__':
    tyro.cli(main)
