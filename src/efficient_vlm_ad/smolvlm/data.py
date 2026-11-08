"""Lazy FP32 views of cached native SmolVLM vision outputs."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from ..hf_data import load_prepared_records
from .cache import FEATURE_FILE, record_key, validate_cache


class CachedVLMDataset(Dataset):
    def __init__(self, cfg, split, max_samples=None, *, processor=None, geometry=None, validate=True):
        self.cfg = cfg
        directory = Path(cfg.cache.dir)
        if validate:
            if processor is None or geometry is None:
                raise ValueError('processor and geometry are required for cache validation')
            manifest = validate_cache(cfg, processor, geometry)
        else:
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        path = Path(cfg.project.output_dir) / 'prepared_data' / f'{split}.jsonl'
        if not path.is_file():
            raise ValueError('Prepared split is missing; prepare data first')
        records = load_prepared_records(cfg, split)
        if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
            raise ValueError('max_samples must be a positive integer')
        self.records = records if max_samples is None else records[:max_samples]
        self.index = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
        self.shape = tuple(manifest['shape'])
        self._features = None
        self.summary = {'split': split, 'num_samples': len(self.records), 'cache_shape': self.shape}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self._features is None:
            self._features = np.memmap(Path(self.cfg.cache.dir) / FEATURE_FILE, mode='r', dtype=np.float16, shape=self.shape)
        record = self.records[idx]
        features = np.array(self._features[self.index[record_key(record, self.cfg.data.view_order)]], dtype=np.float32, copy=True)
        return {'question': record.question, 'answer': record.answer,
                'visual_features': torch.from_numpy(features), 'sample_id': record.sample_id,
                'eval_id': record.eval_id}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_features'] = None
        return state
