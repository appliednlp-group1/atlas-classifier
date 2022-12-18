from transformers import AutoModelForSequenceClassification


def build_classifier(bert_path: str,
                     num_labels: int,
                     ) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(bert_path,
                                                              num_labels=num_labels,
                                                              )
