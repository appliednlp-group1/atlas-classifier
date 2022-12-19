import torch
from datasets import load_dataset
from transformers.models.rag.retrieval_rag import CustomHFIndex

from build_contriever import build_q_encoder, build_q_tokenizer


def run(bert_model: str,
        contriever_model: str,
        contriever_path: str) -> None:
    tokenizer = build_q_tokenizer(contriever_model)
    q_encoder = build_q_encoder(bert_model, contriever_path)
    q_encoder.cuda()
    q_encoder.eval()

    with torch.no_grad():
        dataset = load_dataset('ag_news')
        
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
                )[0].cpu().detach().numpy().tolist()
            return example
        
        dataset = dataset.map(process, batched=True)
    
    index = CustomHFIndex(768,
                          dataset)
    print(index)


if __name__ == '__main__':
    run('bert-base-uncased',
        'facebook/contriever',
        '../models/models/atlas/base/model.pth.tar')
