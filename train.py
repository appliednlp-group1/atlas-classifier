import torch
from build_contriever import build_config, build_q_tokenizer, build_q_encoder, build_retriever
from build_classifier import build_classifier
from atlas_classifier import AtclsModel
from dataloader import build_dataloader

DEFAULT_RETRIEVER_MODEL_PATH = 'facebook/contriever'


def run(bert_model: str,
        contriever_model: str,
        contriever_path: str,
        dataset_path: str,
        batch_size: int,
        ):
    config = build_config(bert_model,
                          contriever_model)
    tokenizer = build_q_tokenizer(contriever_model)
    retriever = build_retriever(bert_model,
                                contriever_model,
                                dataset_path)
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
    
    
