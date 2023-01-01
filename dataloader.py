import os
import json

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def build_dataset(dataset_name: str,
                  phase: str = 'train',
                  use_ratio: float = 1.0,
                  ) -> Dataset:
    assert phase in ('train', 'test')
    assert dataset_name == 'ag_news'
    indices_path = os.path.join(os.path.dirname(__file__), 'indices.json')

    dataset = load_dataset(dataset_name)[phase]
    use_num = int(len(dataset) * use_ratio)
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    indices = indices[:use_num]

    dataset = dataset.filter(lambda x, idx: idx in indices,
                             with_indices=True,
                             )

    return dataset


def build_dataloader(dataset_name: str,
                     tokenizer,
                     batch_size: int,
                     phase: str = 'train',
                     use_ratio: float = 1.0) -> DataLoader:
    dataset = build_dataset(dataset_name, phase, use_ratio)
    
    def get_collater(t):
        def collater(d):
            x = t.batch_encode_plus(
                [r['text'] for r in d],
                padding='longest',
                return_tensors='pt',
                truncation=True,
                )
            y = torch.tensor(
                [r['label'] for r in d]
                ).long()
            return {
                'input_ids': x['input_ids'],
                'attention_mask': x['attention_mask'],
                'label': y,
            }
        return collater
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=phase == 'train',
                      collate_fn=get_collater(tokenizer))


if __name__ == '__main__':
    from build_dpretriever import build_q_tokenizer
    tk = build_q_tokenizer('/app/data/models/dpr_transformers/q_encoder')
    loader = build_dataloader('ag_news',
                              tk,
                              2,
                              'train')
    print(next(iter(loader)))
