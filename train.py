import argparse
import os
import csv
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from build_contriever import build_config, build_q_tokenizer, build_q_encoder, build_retriever
from build_classifier import build_classifier
from atlas_classifier import AtclsModel
from dataloader import build_dataloader

DEFAULT_RETRIEVER_MODEL_PATH = 'facebook/contriever'


def run(bert_model: str,
        contriever_model: str,
        contriever_path: str,
        dataset_path: str,
        index_path: str,
        batch_size: int,
        num_epochs: int,
        lr: float,
        out_dir: str,
        use_ratio: float,
        ):
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    config = build_config(bert_model,
                          contriever_model)
    tokenizer = build_q_tokenizer(contriever_model)
    retriever = build_retriever(bert_model,
                                contriever_model,
                                dataset_path,
                                index_path)
    q_encoder = build_q_encoder(bert_model,
                                contriever_path)
    classifier = build_classifier(bert_model,
                                  num_labels=4)

    if device == 'cuda:0':
        q_encoder = torch.nn.DataParallel(q_encoder)
        classifier = torch.nn.DataParallel(classifier)
        q_encoder.to(device)
        classifier.to(device)

    atcls = AtclsModel(config,
                       q_encoder,
                       classifier,
                       retriever,
                       )
    
    train_loader = build_dataloader('ag_news',
                                    tokenizer,
                                    batch_size,
                                    phase='train',
                                    use_ratio=use_ratio)
    test_loader = build_dataloader('ag_news',
                                   tokenizer,
                                   batch_size,
                                   phase='test',
                                   use_ratio=use_ratio)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(atcls.parameters(), lr=lr)

    if device == 'cuda':
        atcls = torch.nn.DataParallel(atcls)  # make parallel
        atcls.to(device)
        torch.backends.cudnn.benchmark = True
    
    with open(os.path.join(out_dir, 'result.csv'), 'a') as f:
        csv.writer(f).writerow([
            'dt',
            'epoch',
            'train_loss',
            'test_loss',
            'train_acc',
            'test_acc'])
        
    for epoch in range(1, num_epochs + 1):
        atcls.train()
        
        train_total = 0
        train_corrects = 0
        train_losses = []
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids, attention_mask, label = (
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['label'].to(device),
                    )
            out = atcls(input_ids,
                        attention_mask)
            pred = out['logits']
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            _, y = torch.max(pred.cpu(), 1)
            train_total += len(batch['label'])
            train_corrects += (y == batch['label']).sum().item()
            
        atcls.eval()
        
        test_total = 0
        test_corrects = 0
        test_losses = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                input_ids, attention_mask, label = (
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device),
                        batch['label'].to(device),
                        )
                out = atcls(input_ids,
                            attention_mask)
                pred = out['logits']
                loss = criterion(pred, label)
                
                test_losses.append(loss.item())
                
                _, y = torch.max(pred.cpu(), 1)
                test_total += len(batch['label'])
                test_corrects += (y == batch['label']).sum().item()

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        train_acc = train_corrects / train_total * 100
        test_acc = test_corrects / test_total * 100
        
        print(f'[Epoch {epoch:04}]: '
              f'train loss {train_loss:.3f} / '
              f'test loss {test_loss:.3f} / '
              f'train acc {train_acc:.2f} / '
              f'test acc {test_acc:.2f}')
        
        with open(os.path.join(out_dir, 'result.csv'), 'a') as f:
            csv.writer(f).writerow([
                datetime.now(),
                epoch,
                train_loss,
                test_loss,
                train_acc,
                test_acc])
        
        epoch_dir = os.path.join(out_dir, f'acc{test_acc:.2f}_e{epoch}')
        epoch_tok_dir = os.path.join(epoch_dir, 'tokenizer')
        os.makedirs(epoch_tok_dir, exist_ok=True)
        tokenizer.save_pretrained(epoch_tok_dir)
        epoch_mod_dir = os.path.join(epoch_dir, 'model')
        os.makedirs(epoch_mod_dir, exist_ok=True)
        atcls.save_pretrained(epoch_mod_dir)

    last_dir = os.path.join(out_dir, f'elast')
    last_tok_dir = os.path.join(last_dir, 'tokenizer')
    os.makedirs(last_tok_dir, exist_ok=True)
    tokenizer.save_pretrained(last_tok_dir)
    last_mod_dir = os.path.join(last_dir, 'model')
    os.makedirs(last_mod_dir, exist_ok=True)
    atcls.save_pretrained(last_mod_dir)


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
                        default=64)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50)
    parser.add_argument('--lr',
                        type=float,
                        default=0.00001)
    parser.add_argument('--out_dir',
                        type=str,
                        default='results/test1')
    parser.add_argument('--use_ratio',
                        type=str,
                        default=0.01)
    
    args = parser.parse_args()
    
    run(args.bert_model,
        args.contriever_model,
        args.contriever_path,
        args.dataset_path,
        args.index_path,
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.out_dir,
        args.use_ratio)

    
