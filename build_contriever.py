import torch
import transformers
from transformers.models.rag.retrieval_rag import CustomHFIndex
from datasets import load_from_disk


def build_q_tokenizer(contriever_model: str) -> transformers.AutoTokenizer:
    return transformers.AutoTokenizer.from_pretrained(contriever_model)


def _build_g_tokenizer(bert_model: str) -> transformers.BertTokenizer:
    return transformers.BertTokenizer.from_pretrained(bert_model)


def build_config(bert_model: str,
                 contriever_model: str) -> transformers.RagConfig:
    q_encoder = transformers.AutoModel.from_pretrained(contriever_model)
    classifier = transformers.AutoModel.from_pretrained(bert_model)
    return transformers.RagConfig(question_encoder=q_encoder.config.to_dict(),
                                  generator=classifier.config.to_dict())


def _build_index(hidden_dim: int,
                 dataset_path: str,
                 index_path: str,
                 ) -> CustomHFIndex:
    return CustomHFIndex.load_from_disk(hidden_dim,
                                        dataset_path,
                                        index_path)


def build_q_encoder(bert_model: str,
                    contriever_path: str) -> transformers.BertModel:
    checkpoint = torch.load(contriever_path, map_location='cpu')
    q_encoder = transformers.BertModel.from_pretrained(bert_model)
    q_encoder.embeddings.load_state_dict({
        k.replace('retriever.module.contriever.embeddings.', ''): v
        for k, v in checkpoint['model'].items()
        if k.startswith('retriever.module.contriever.embeddings.')
        })
    q_encoder.encoder.load_state_dict({
        k.replace('retriever.module.contriever.encoder.', ''): v
        for k, v in checkpoint['model'].items()
        if k.startswith('retriever.module.contriever.encoder.')
        })
    return q_encoder


def build_retriever(bert_model: str,
                    contriever_model: str,
                    dataset_path: str,
                    index_path: str,
                    hidden_dim: int = 786,
                    ) -> transformers.RagRetriever:
    config = build_config(bert_model,
                          contriever_model)
    q_tokenizer = build_q_tokenizer(contriever_model)
    g_tokenizer = _build_g_tokenizer(bert_model)
    index = _build_index(hidden_dim,
                         dataset_path,
                         index_path)
    return transformers.RagRetriever(config,
                                     q_tokenizer,
                                     g_tokenizer,
                                     index=index)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    opt = {
        'bert_model': 'bert-base-uncased',
        'contriever_model': 'facebook/contriever',
        'contriever_path': '../models/models/atlas/base/model.pth.tar',
        'dataset_path': '../data/datasets/train',
        'index_path': '../data/datasets/agnews.faiss',
    }
    tokenizer = build_q_tokenizer(opt['contriever_model'])
    retriever = build_retriever(opt['bert_model'],
                                opt['contriever_model'],
                                opt['dataset_path'],
                                opt['index_path'])
    q_encoder = build_q_encoder(opt['bert_model'],
                                opt['contriever_path']).to(device)

    inputs = tokenizer('The stock has fallen today.')
    enc_outputs = q_encoder(torch.Tensor([inputs['input_ids']]).long().to(device),
                            attention_mask=torch.Tensor([inputs['attention_mask']]).to(device),
                            return_dict=True)
    out = retriever(torch.Tensor([inputs['input_ids']]).long(),
                    enc_outputs[1].cpu().clone().detach().to(torch.float32).numpy(),
                    n_docs=5,
                    return_tensors='pt')
    ans = tokenizer.batch_decode(out['context_input_ids'],
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    for a in ans:
        print(a)
