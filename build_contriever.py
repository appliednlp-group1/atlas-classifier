import torch
import transformers


def build_q_tokenizer(contriever_model: str) -> transformers.AutoTokenizer:
    return transformers.AutoTokenizer.from_pretrained(contriever_model)


def _build_g_tokenizer(bert_model: str) -> transformers.BertTokenizer:
    return transformers.BertTokenizer.from_pretrained(bert_model)


def build_config(bert_model: str,
                 contriever_model: str) -> transformers.RagConfig:
    q_encoder = transformers.AutoConfig.from_pretrained(contriever_model)
    classifier = transformers.AutoConfig.from_pretrained(bert_model)
    return transformers.RagConfig(question_encoder=q_encoder.config.to_dict(),
                                  generator=classifier.config.to_dict())


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
