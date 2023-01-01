import os.path

import torch

from dataloader import build_dataset
from build_contriever import build_q_encoder, build_q_tokenizer


def run(bert_model: str,
        contriever_model: str,
        contriever_path: str,
        index_dir: str,
        batch_size: int = 128,
        use_ratio: float = 1.0,
        ) -> None:
    out_dir = os.path.join(index_dir, f'r{use_ratio}')
    os.makedirs(out_dir, exist_ok=True)
    
    tokenizer = build_q_tokenizer(contriever_model)
    q_encoder = build_q_encoder(bert_model, contriever_path)
    q_encoder.cuda()
    q_encoder.eval()

    with torch.no_grad():
        dataset = build_dataset('ag_news', phase='train', use_ratio=use_ratio)
        
        def process(example):
            inputs = tokenizer(example['text'],
                               truncation=True,
                               pad_to_max_length=True,
                               return_tensors='pt',
                               )
            example['embeddings'] = q_encoder(
                input_ids=torch.tensor(inputs['input_ids']).long().cuda(),
                attention_mask=torch.tensor(inputs['attention_mask']).cuda(),
                return_dict=True,
                )[1].cpu().clone().detach().numpy().tolist()
            example['title'] = ['']*len(example['label'])
            return example
        
        dataset = dataset.map(process, batched=True, batch_size=batch_size)
    
    dataset.save_to_disk(os.path.join(out_dir, 'datasets'))
    
    ds = dataset['train']
    ds = ds.add_faiss_index('embeddings', index_name='agnews')
    
    ds.save_faiss_index('agnews', os.path.join(out_dir, 'agnews.faiss'))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ratio',
                        type=float,
                        default=0.01)
    args = parser.parse_args()
    
    run('bert-base-uncased',
        'facebook/contriever',
        '../models/models/atlas/base/model.pth.tar',
        '../data',
        use_ratio=args.use_ratio)
