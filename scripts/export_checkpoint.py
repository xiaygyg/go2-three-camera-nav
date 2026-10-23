"""Export a portable navigation checkpoint with an allowlist of public metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', default='assets/checkpoints/go2_three_camera.pt')
    args = parser.parse_args()
    payload = torch.load(args.source, map_location='cpu', weights_only=True)
    keys = ['sensor_mode', 'camera_config', 'scan_column0', 'dynamic_source', 'robot',
            'lower_policy', 'lower_policy_commit', 'lower_proprioception_dim', 'lower_history_shape',
            'training_supervisor_enabled', 'supervisor_revision', 'yaw_mode', 'recovery_schema',
            'state_index_2', 'state_index_4', 'execution', 'novelty_reward_episode_cap',
            'phase', 'seed', 'physics_transitions', 'action_low', 'action_high', 'state_dim']
    metadata = {k: payload['metadata'][k] for k in keys if k in payload['metadata']}
    if metadata.get('robot') != 'go2':
        raise ValueError('Expected Go2 navigation weights')
    result = Path(args.output)
    if result.exists():
        raise FileExistsError('Use a new checkpoint output')
    result.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': payload['model'], 'metadata': metadata}, result)
    verified = torch.load(result, weights_only=True, map_location='cpu')
    assert all(torch.equal(v, verified['model'][k]) for k, v in payload['model'].items())
    result.with_suffix('.json').write_text(json.dumps({
        'sha256': hashlib.sha256(result.read_bytes()).hexdigest(),
        'tensor_values_preserved': True, 'metadata': metadata}, indent=2))
    print('Exported public Go2 checkpoint; tensor values preserved.')


if __name__ == '__main__':
    main()
