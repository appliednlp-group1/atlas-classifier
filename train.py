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


def get_lr_func(num_epochs: int):
    def lr_func(epoch):
        if epoch < num_epochs * 0.8:
            return 1.0
        else:
            return 0.1
    return lr_func


def run(bert_model: str,
        contriever_model: str,
        contriever_path: str,
        dataset_path: str,
        index_path: str,
        batch_size: int,
        num_epochs: int,
        lr: float,
        out_dir: str,
        ):
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    atcls = AtclsModel(config,
                       q_encoder,
                       classifier,
                       retriever,
                       )
    
    train_loader = build_dataloader('ag_news',
                                    tokenizer,
                                    batch_size,
                                    phase='train')
    test_loader = build_dataloader('ag_news',
                                   tokenizer,
                                   batch_size,
                                   phase='test')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(atcls.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     get_lr_func(num_epochs))
    
    atcls.to(device)
    
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
        for batch in tqdm(train_loader):
            out = atcls(batch['input_ids'].to(device),
                        batch['attention_mask'].to(device))
            pred = out['logits']
            optimizer.zero_grad()
            loss = criterion(pred, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            _, y = torch.max(pred.cpu(), 1)
            train_total += len(batch)
            train_corrects += (y == batch['label']).sum().item()
            
        atcls.eval()
        
        test_total = 0
        test_corrects = 0
        test_losses = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                out = atcls(batch['input_ids'].to(device),
                            batch['attention_mask'].to(device))
                pred = out['logits']
                loss = criterion(pred, batch['label'].to(device))
                
                test_losses.append(loss.item())
                
                _, y = torch.max(pred.cpu(), 1)
                test_total += len(batch)
                test_corrects += (y == batch['label']).sum().item()

        lr_scheduler.step()
        
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
        os.makedirs(epoch_dir, exist_ok=True)
        tokenizer.save_pretrained(epoch_dir)
        atcls.save_pretrained(epoch_dir)

    last_dir = os.path.join(out_dir, f'elast')
    os.makedirs(last_dir, exist_ok=True)
    tokenizer.save_pretrained(last_dir)
    atcls.save_pretrained(last_dir)


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
                        default='../models/models/atlas/base/model.pth.tar')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='../data/datasets/train')
    parser.add_argument('--index_path',
                        type=str,
                        default='../data/datasets/agnews.faiss')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=2)
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001)
    parser.add_argument('--out_dir',
                        type=str,
                        default='../results/test1')
    
    args = parser.parse_args()
    
    run(args.bert_model,
        args.contriever_model,
        args.contriever_path,
        args.dataset_path,
        args.index_path,
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.out_dir)
    
