import argparse
import os
import json

import torch
import transformers
from transformers import BertModel, BertForSequenceClassification
from tqdm import tqdm

from dataloader import build_dataloader
from build_contriever import build_retriever, build_config, build_q_encoder
from atlas_classifier import forward


def run(out_dir: str,
        bert_model: str,
        contriever_model: str,
        contriever_path: str,
        dataset_path: str,
        index_path: str,
        batch_size: int,
        use_ratio: float,
        no_retriever: bool,
        ):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(out_dir, 'tokenizer'))
    # q_encoder = build_q_encoder(bert_model,
    #                             contriever_path,
    #                             )
    q_encoder = BertModel.from_pretrained(os.path.join(out_dir, 'model/q_encoder'))
    classifier = BertForSequenceClassification.from_pretrained(os.path.join(out_dir, 'model/classifier'))
    
    if device == 'cuda:0':
        q_encoder = torch.nn.DataParallel(q_encoder)
        classifier = torch.nn.DataParallel(classifier)
        q_encoder.to(device)
        classifier.to(device)
        torch.backends.cudnn.benchmark = True
        
    config = build_config(bert_model,
                          contriever_model,
                          )
    retriever = build_retriever(bert_model,
                                contriever_model,
                                dataset_path,
                                index_path,
                                )

    test_loader = build_dataloader('ag_news',
                                   tokenizer,
                                   batch_size,
                                   phase='test',
                                   use_ratio=use_ratio)
    
    q_encoder.eval()
    classifier.eval()
    
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids, attention_mask, label = (
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['label'].to(device),
            )
            out = forward(input_ids,
                          attention_mask,
                          q_encoder,
                          retriever,
                          classifier,
                          n_docs=config.n_docs,
                          output_attentions=config.output_attentions,
                          no_retriever=no_retriever,
                          )
            source_texts = tokenizer.batch_decode(input_ids,
                                                  skip_special_tokens=True,
                                                  )
            doc_scores = out['doc_scores'].cpu().clone().detach().numpy()
            context_input_ids = out['context_input_ids'].cpu().clone().detach()
            for i in range(batch_size):
                source_text = source_texts[i]
                doc_score = doc_scores[i]
                target_texts = tokenizer.batch_decode(
                    context_input_ids[config.n_docs*i:config.n_docs*(1+i), :],
                    skip_special_tokens=True,
                    )
                target_texts = [t.split(r'/ ')[1].split(r' / / ')[0] for t in target_texts]
                results.append({
                    'source_text': source_text,
                    'targets': [
                        {
                            'score': float(doc_score[j]),
                            'text': target_texts[j],
                        }
                        for j in range(config.n_docs)
                    ]
                })
    
    with open(os.path.join(out_dir, 'inference.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model',
                        type=str,
                        default='bert-base-uncased')
    parser.add_argument('--contriever_model',
                        type=str,
                        default='facebook/contriever')
    parser.add_argument('--contriever_path',
                        type=str,
                        default='models/models/atlas/base/model.pth.tar')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='data/datasets/train')
    parser.add_argument('--index_path',
                        type=str,
                        default='data/datasets/agnews.faiss')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--out_dir',
                        type=str,
                        default='results/test1')
    parser.add_argument('--use_ratio',
                        type=str,
                        default=0.1)
    parser.add_argument('--no_retriever',
                        action='store_true')
    
    args = parser.parse_args()
    
    run(args.out_dir,
        args.bert_model,
        args.contriever_model,
        args.contriever_path,
        args.dataset_path,
        args.index_path,
        args.batch_size,
        args.use_ratio,
        args.no_retriever,
        )
