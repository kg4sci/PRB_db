"""Offline integration checks for the packaged experiment entry points."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ANTHROPIC = ROOT / 'inference/experiments/anthropic'


class ExperimentReleaseTests(unittest.TestCase):
    def test_all_entrypoints_use_selected_inputs_and_resume(self):
        files = sorted((ROOT / 'inference/experiments').glob('*/inference_single_stage*.py'))
        self.assertEqual(len(files), 22)
        fixtures = {}
        for name in ['benchmark2025.json', 'benchmark2025_keyword_noise_mixed.json',
                     'benchmark2025_synthetic_degradation.json']:
            fixtures[name] = json.loads((ROOT / 'dataset' / name).read_text())[0]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'LLM_API_KEY': 'offline-test', 'LLM_BASE_URL': 'http://localhost:1',
            'LLM_MODEL_NAME': 'offline-model', 'INFERENCE_WRITE_FINAL_JSON': '0',
        }), patch('socket.socket.connect', side_effect=AssertionError('Network is disabled')):
            self.assertFalse((ANTHROPIC / "anthropic_inference_common.py").exists())
            for index, file in enumerate(files):
                with self.subTest(entrypoint=str(file.relative_to(ROOT))):
                    spec = importlib.util.spec_from_file_location(f'offline_entry_{index}', file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    name = 'benchmark2025.json'
                    for condition in ['keyword_noise_mixed', 'synthetic_degradation']:
                        if condition in file.name:
                            name = f'benchmark2025_{condition}.json'
                    entry = fixtures[name]
                    dataset = Path(temporary) / f'input{index}.json'
                    dataset.write_text(json.dumps([entry]))
                    output = Path(temporary) / f'result{index}.json'
                    abstract = entry.get('degraded_abstract_text', entry['paper_content']['abstract_text'])
                    calls = []

                    def fake_call(messages, **kwargs):
                        calls.append(messages)
                        user = messages[-1]['content']
                        self.assertIn(abstract, user)
                        stress = name != 'benchmark2025.json'
                        self.assertEqual('POST-DECISION TRANSFER OPTIONS' in user, not stress)
                        if name.endswith('keyword_noise_mixed.json'):
                            self.assertIn(entry['paper_content']['keywords'], user)
                        return {'Primary_Decision': entry['ground_truth']['expected_decision'],
                                'Quality_Assessment': 'Exceeds Journal Caliber',
                                'Rejection_Category': 'Insufficient Novelty/Impact',
                                'Confidence_Score': 8, 'Transfer_Ranking': None}, False

                    args = dict(dataset_path=str(dataset), output_path=str(output), inference_num=1)
                    signature = inspect.signature(module.run_inference)
                    if 'system_prompt_builder' in signature.parameters:
                        args['system_prompt_builder'] = module.create_system_prompt
                    if 'user_prompt_builder' in signature.parameters and hasattr(module, 'create_user_prompt'):
                        args['user_prompt_builder'] = module.create_user_prompt
                    if 'expects_reasoning_process' in signature.parameters:
                        args['expects_reasoning_process'] = '_CoT' in file.name
                    if 'progress_desc' in signature.parameters:
                        args['progress_desc'] = 'Offline validation'
                    with patch.dict(module.run_inference.__globals__, {'_call_model': fake_call}):
                        module.run_inference(**args)
                        module.run_inference(**args)
                    self.assertEqual(len(calls), 1, 'A completed sample should resume without another call')
                    results = [json.loads(line) for line in Path(str(output)+'.jsonl').read_text().splitlines()]
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0]['uid'], entry['uid'])
                    self.assertEqual(results[0]['ground_truth'], entry['ground_truth'])


if __name__ == '__main__':
    unittest.main()
