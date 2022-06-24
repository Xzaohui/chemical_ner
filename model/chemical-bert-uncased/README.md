---
language: "en"
tags:
- chemical-domain
- safety-datasheets
widget:
- text: "The removal of mercaptans, and for drying of gases and [MASK]."
---
# BERT for Chemical Industry
A BERT-based language model further pre-trained from the checkpoint of [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased). We used a corpus of over 40,000+ technical documents from the **Chemical Industrial domain** and combined it with 13,000 Wikipedia Chemistry articles, ranging from Safety Data Sheets and Products Information Documents, with 250,000+ tokens from the Chemical domain and pre-trained using MLM and over 9.2 million paragraphs.
- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT internally masks the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
```python
from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="recobo/chemical-bert-uncased",
    tokenizer="recobo/chemical-bert-uncased"
)
fill_mask("we create [MASK]")
```