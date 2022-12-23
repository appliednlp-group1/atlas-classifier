import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def build_dataloader(dataset_name: str,
                     tokenizer,
                     batch_size: int,
                     phase: str = 'train',
                     use_ratio: float = 1.0) -> DataLoader:
    assert phase in ('train', 'test')
    dataset = load_dataset(dataset_name)[phase]
    use_num = int(len(dataset) * use_ratio)
    dataset, _ = torch.utils.data.random_split(dataset, [use_num, len(dataset) - use_num])
    
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
