from typing import Dict, Optional, Any

import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.rag.retrieval_rag import RagRetriever, RagConfig


class AtclsPreTrainedModel(PreTrainedModel):
    config_class = RagConfig
    base_model_prefix = 'atcls'
    _keys_to_ignore_on_load_missing = ['position_ids']
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs['_fast_init'] = False
        return super().from_pretrained(*args, **kwargs)


class AtclsModel(AtclsPreTrainedModel):
    def __init__(self,
                 config: PretrainedConfig,
                 question_encoder: PreTrainedModel,
                 classifier: Any,
                 retriever: RagRetriever,
                 ):
        super().__init__(config)
        
        self.question_encoder = question_encoder
        self.classifier = classifier
        self.retriever = retriever
        
        self.n_docs = config.n_docs
        self.output_attentions = config.output_attentions
        self.output_retrieved = config.output_retrieved
        
    def forward(self,
                input_ids: Optional[torch.LongTensor],
                attention_mask: Optional[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        question_enc_outputs = self.question_encoder(input_ids,
                                                     attention_mask=attention_mask,
                                                     return_dict=True)
        question_encoder_last_hidden_state = question_enc_outputs[0]
        
        retriever_outputs = self.retriever(input_ids,
                                           question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                                           prefix=self.retriever.config.prefix,
                                           n_docs=self.n_docs,
                                           return_tensors='pt')
        
        context_input_ids = retriever_outputs['context_input_ids'].to(input_ids)
        context_attention_mask = retriever_outputs['context_attention_mask'].to(input_ids)
        retrieved_doc_embeds = retriever_outputs['retrieved_doc_embeds'].to(question_encoder_last_hidden_state)
        retrieved_doc_ids = retriever_outputs['doc_ids']
        
        doc_scores = torch.bmm(question_encoder_last_hidden_state.unsqueeze(1),
                               retrieved_doc_embeds.transpose(1, 2),
                               ).squeeze(1)
        
        cls_outputs = self.classifier(input_ids=context_input_ids,
                                      attention_mask=context_attention_mask,
                                      output_attentions=self.output_attentions,
                                      return_dict=True)
        cls_logits = cls_outputs.logits
        
        logits = (
            (doc_scores.unsqueeze(2) * cls_logits.view(*doc_scores.shape, cls_logits.shape[1])).sum(axis=1)
            / (doc_scores.unsqueeze(2).sum(axis=1))
        )
        
        return {
            'doc_scores': doc_scores,
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'retrieved_doc_embeds': retrieved_doc_embeds,
            'retrieved_doc_ids': retrieved_doc_ids,
            'question_enc_last_hidden_state': question_encoder_last_hidden_state,
            'question_enc_hidden_states': question_enc_outputs.hidden_states,
            'question_enc_attentions': question_enc_outputs.attentions,
            'classifier_logits': cls_logits,
            'logits': logits,
        }


if __name__ == '__main__':
    from build_classifier import build_classifier
    from build_dpretriever import build_config, build_retriever, build_q_encoder, build_q_tokenizer
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    opt = {
        'question_model': '/app/data/models/dpr_transformers/q_encoder',
        'indexdata_path': '/app/data/dataset/dpr_knowledge_index/knowledge',
        'index_path': '/app/data/dataset/dpr_knowledge_index/knowledge_index.faiss',
        'bert_path': '/app/data/models/small_bert',
    }
    
    atcls = AtclsModel(build_config(opt['question_model']),
                       build_q_encoder(opt['question_model']),
                       build_classifier(opt['bert_path'], 3),
                       build_retriever(opt['question_model'],
                                       opt['indexdata_path'],
                                       opt['index_path']),
                       ).to(device)
    tokenizer = build_q_tokenizer(opt['question_model'])
    
    inputs = tokenizer('2005年から2015年で埼玉県の人口は変わっていますか？')
    ids = torch.Tensor([inputs['input_ids'] for _ in range(2)]).long().to(device)
    mask = torch.Tensor([inputs['attention_mask'] for _ in range(2)]).to(device)
    
    out = atcls(ids,
                mask)

    print(out['logits'])
