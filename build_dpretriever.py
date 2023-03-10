import transformers
from transformers.models.rag.retrieval_rag import CustomHFIndex


def build_q_tokenizer(question_model: str) -> transformers.BertTokenizer:
    return transformers.BertJapaneseTokenizer.from_pretrained(question_model)


def _build_g_tokenizer(question_model: str) -> transformers.BertTokenizer:
    return transformers.BertJapaneseTokenizer.from_pretrained(question_model)


def build_config(question_model: str) -> transformers.RagConfig:
    q_enc_config = transformers.DPRConfig.from_pretrained(question_model)
    bert_config = transformers.BertConfig.from_pretrained(question_model)
    return transformers.RagConfig(question_encoder=q_enc_config.to_dict(),
                                  generator=bert_config.to_dict())


def _build_index(
        hidden_dim: int,
        dataset_path: str,
        index_path: str,
        ) -> CustomHFIndex:
    return CustomHFIndex.load_from_disk(
        hidden_dim,
        dataset_path=dataset_path,
        index_path=index_path)


def build_q_encoder(question_model: str) -> transformers.DPRQuestionEncoder:
    return transformers.DPRQuestionEncoder.from_pretrained(question_model)


def build_retriever(
        question_model: str,
        indexdata_path: str,
        index_path: str,
        hidden_dim: int = 786,
        ) -> transformers.RagRetriever:
    config = build_config(question_model)
    q_tokenizer = build_q_tokenizer(question_model)
    g_tokenizer = _build_g_tokenizer(question_model)
    index = _build_index(hidden_dim, indexdata_path, index_path)
    return transformers.RagRetriever(config,
                                     q_tokenizer,
                                     g_tokenizer,
                                     index=index)


if __name__ == '__main__':
    import torch
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    opt = {
        'question_model': '/app/data/models/dpr_transformers/q_encoder',
        'indexdata_path': '/app/data/dataset/dpr_knowledge_index/knowledge',
        'index_path': '/app/data/dataset/dpr_knowledge_index/knowledge_index.faiss',
    }
    tokenizer = build_q_tokenizer(opt['question_model'])
    retriever = build_retriever(opt['question_model'],
                                opt['indexdata_path'],
                                opt['index_path'])
    q_encoder = build_q_encoder(opt['question_model']).to(device)
    
    inputs = tokenizer('2005?????????2015??????????????????????????????????????????????????????')
    enc_outputs = q_encoder(torch.Tensor([inputs['input_ids']]).long().to(device),
                            attention_mask=torch.Tensor([inputs['attention_mask']]).to(device),
                            return_dict=True)
    out = retriever(torch.Tensor([inputs['input_ids']]).long(),
                    enc_outputs[0].cpu().clone().detach().to(torch.float32).numpy(),
                    n_docs=5,
                    return_tensors='pt')
    ans = tokenizer.batch_decode(out['context_input_ids'],
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    for a in ans:
        print(a)
