"""Launch one released paper experiment without changing its inference logic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    'main': ('', 'benchmark2025.json'),
    'no_metrics': ('_no_journal_metrics', 'benchmark2025.json'),
    'name_only': ('_name_only', 'benchmark2025.json'),
    'keyword_noise': ('_real_submission_keyword_noise_mixed', 'benchmark2025_keyword_noise_mixed.json'),
    'synthetic_degradation': ('_real_submission_synthetic_degradation', 'benchmark2025_synthetic_degradation.json'),
    'send90hint': ('_send90hint', 'benchmark2025.json'),
}


def build_command(args: argparse.Namespace) -> list[str]:
    if args.backend == 'anthropic' and args.condition == 'send90hint':
        raise ValueError('The paper reports send90hint only for GPT through the OpenAI-compatible backend.')
    if args.num is not None and args.num < 0:
        raise ValueError('--num must be nonnegative')
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Require --num-shards >= 1 and 0 <= --shard-id < --num-shards')
    suffix, dataset = CONDITIONS[args.condition]
    mode = '' if args.prompt == 'direct' else '_CoT'
    script = ROOT / 'inference/experiments' / args.backend / f'inference_single_stage{mode}{suffix}.py'
    if not script.is_file():
        raise ValueError(f'Missing experiment entry point: {script}')
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else ROOT / 'dataset' / dataset
    if not dataset_path.is_file():
        raise ValueError(f'Missing dataset: {dataset_path}')
    output = Path(args.output).expanduser().resolve()
    command = [sys.executable, str(script), '--dataset', str(dataset_path), '--output', str(output),
               '--num-shards', str(args.num_shards), '--shard-id', str(args.shard_id)]
    for flag, value in [('--num', args.num), ('--uid', args.uid), ('--sample-id', args.sample_id),
                        ('--finished-from', args.finished_from)]:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--prompt', choices=['direct', 'rationale_first'], default='direct')
    parser.add_argument('--backend', choices=['openai', 'anthropic'], default='openai',
                        help='API protocol, not model family; anthropic expects a base URL without /v1/messages.')
    parser.add_argument('--model', help='Provider model identifier; overrides LLM_MODEL_NAME.')
    parser.add_argument('--dataset', help='Optional dataset override; defaults to the released file for this condition.')
    parser.add_argument('--output', required=True, help='Output base path; scripts append .jsonl by default.')
    parser.add_argument('--num', type=int)
    parser.add_argument('--uid')
    parser.add_argument('--sample-id')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--finished-from')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without importing inference code or contacting an API.')
    args = parser.parse_args()
    try:
        command = build_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    env = os.environ.copy()
    if args.model:
        env['LLM_MODEL_NAME'] = args.model
    if args.dry_run:
        print(json.dumps({'backend': args.backend, 'condition': args.condition,
                          'prompt': args.prompt, 'model_override': args.model,
                          'command': shlex.join(command)}, indent=2))
        return 0
    prefix = 'ANTHROPIC' if args.backend == 'anthropic' else 'OPENAI'
    required = [('LLM_API_KEY', f'{prefix}_API_KEY'), ('LLM_BASE_URL', f'{prefix}_BASE_URL'),
                ('LLM_MODEL_NAME', f'{prefix}_MODEL')]
    missing = [' / '.join(names) for names in required if not any(env.get(name) for name in names)]
    if missing:
        parser.error('Set the following environment variables (or --model): ' + ', '.join(missing))
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
