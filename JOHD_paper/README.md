# Neural Language Models for Nineteenth-Century English: preprocessing and hyperparameters

Here, we provide additional information on the neural language models for nineteenth-century English, published (as of Sep. 2nd, 2021, the data paper is *under-review*) in the Journal of Open Humanities Data (JOHD). This includes preprocessing steps, hyperparameters and other information required to reproduce the models.

## Word2vec

We trained the word2vec (Mikolov et al., 2013) models as implemented in the Gensim library (Rehurek & Sojka, 2011). 
There are two word2vec models:

### `w2v_1760_1900`

We trained this model instance using the whole dataset with the following hyperparameters:

```python
w2v_args = Namespace(
    make_lower_case=True,                  # make lower-casse before training
    remove_words_with_numbers=False,       # remove words with numbers
    remove_puncs=False,                    # remove punctuations
    remove_stop=False,                     # remove stop words
    min_sentence_len=5,                    # sentences with < min_sentence_length will be skipped
    size=300,                              # Dimensionality of word embeddings
    alpha=0.03,                            # The initial learning rate.
    min_alpha=0.0007,                      # Learning rate will linearly drop to min_alpha as training progresses.
    sg=1,                                  # Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs=0,                                  # If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    negative=20,                           # If > 0, negative sampling will be used, the int for negative specifies how many �~@~\noise words�~@~] should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    epochs=1,
    min_count=20,                          # Ignore words that appear less than this
    window=5,                              # Context window for words during training
    sample=1e-3,                           # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    workers=16,                             # Number of processors (parallelisation)
    cbow_mean=1,                           # If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    null_word=0,
    trim_rule=None,                        # Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
    sorted_vocab=1,                        # If 1, sort the vocabulary by descending frequency before assigning word indexes
    batch_words=10000,                     # Target size (in words) for batches of examples passed to worker threads (and thus cython routines).(Larger batches will be passed if individual texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
    compute_loss=True,                     # If True, computes and stores loss value which can be retrieved using get_latest_training_loss().
    seed=1364,
)
```

### `w2v_1760_1850`

We trained this model instance on text published before 1850. The hyperparameters are the same as [w2v_1760_1900](#w2v_1760_1900). The only difference is the input text data.

## fastText

We trained the fastText (Bojanowski et al., 2016) models as implemented in the Gensim library (Rehurek & Sojka, 2011). 
There are two fastText models:

### `ft_1760_1900`

We trained this model instance using the whole dataset with the following hyperparameters:

```python
fasttext_args = Namespace(
    make_lower_case=True,                  # make lower-casse before training
    remove_words_with_numbers=False,       # remove words with numbers
    remove_puncs=False,                    # remove punctuations
    remove_stop=False,                     # remove stop words
    min_sentence_len=5,                    # sentences with < min_sentence_length will be skipped
    size=300,                              # Dimensionality of the word vectors.
    alpha=0.03,                            # The initial learning rate.
    min_alpha=0.0007,                      # Learning rate will linearly drop to min_alpha as training progresses.
    sg=1,                                  # Training algorithm: skip-gram if sg=1, otherwise CBOW.
    hs=0,                                  # If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
    negative=20,                           # If > 0, negative sampling will be used, the int for negative specifies how many �~@~\noise words�~@~] should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    epochs=1,
    min_count=20,                          # The model ignores all words with total frequency lower than this.
    window=5,                              # The maximum distance between the current and predicted word within a sentence.
    sample=1e-3,                           # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    workers=16,
    cbow_mean=1,                           # If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    null_word=0,                           #
    trim_rule=None,                        #
    sorted_vocab=1,                        # If 1, sort the vocabulary by descending frequency before assigning word indices.
    batch_words=10000,                     # Target size (in words) for batches of examples passed to worker threads (and thus cython routines).(Larger batches will be passed if individual texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
    seed=1364,                             # Seed for the random number generator.
    word_ngrams=1,                         # If 1, uses enriches word vectors with subword(n-grams) information. If 0, this is equivalent to Word2Vec.
    min_n=2,                               # Minimum length of char n-grams to be used for training word representations.
    max_n=5,                               # Max length of char ngrams to be used for training word representations. Set max_n to be lesser than min_n to avoid char ngrams being used.
    bucket=2000000)                        # Character ngrams are hashed into a fixed number of buckets, in order to limit the memory usage of the model. This option specifies the number of buckets used by the model.
```

### `ft_1760_1850`

We trained this model instance on text published before 1850. The hyperparameters are the same as [ft_1760_1900](#ft_1760_1900). The only difference is the input text data.


## Flair

Flair is a character language model based on the Long short-term memory (LSTM) variant of recurrent neural networks (Akbik et al., 2019;  Hochreiter & Schmidhuber, 1997). We trained this model instance using the whole dataset with the following hyperparameters:

```python
flair_args = Namespace(
    workers=4,
    epochs=1,
    sequence_length=250,
    is_forward_lm=True,                    # forward or backward Language Model?
    is_character_level=True,
    mini_batch_size=100,
    hidden_size=2048,
    nlayers=1,
    random_seed=1364
    )

# Dictionary
# load the default character dictionary
from flair.data import Dictionary
Dictionary = Dictionary.load('chars')
```

For training, we used one NVIDIA Tesla K80 GPUs, and it took 533.6 GPU hours for one epoch (We trained this model instance using the whole dataset).

## BERT

To fine-tune BERT model instances, we started with a contemporary model: `BERT base uncased` (https://github.com/google-research/bert), hereinafter referred to as *BERT-base* (Devlin, Chang, Lee, & Toutanova, 2019; Wolf et al., 2019). This instance was then fine-tuned on the earliest time period (i.e., books predating 1850). For the consecutive period (1850-1875), we used the pre-1850 language model instance as a starting point and continued fine-tuning with texts from the following period. This procedure of consecutive incremental fine-tuning was repeated for the other two time periods.  

We used the original BERT-base tokenizer as implemented by [HuggingFace](https://github.com/huggingface/transformers) (Wolf et al., 2019). We did not train new tokenizers for each time period. This way, the resulting language model instances can be compared easily with no further processing or adjustments. The tokenized and lowercased sentences were fed to the language model fine-tuning tool in which only the masked language model (MLM) objective was optimized. We used a batch size of 5 per GPU and fine-tuned for 1 epoch over the books in each time-period. The choice of batch size was dictated by the available GPU memory (we used 4x NVIDIA Tesla K80 GPUs in parallel). Similar to the original BERT pre-training procedure, we used the Adam optimizer  (Kingma & Ba, 2014) with a learning rate of 0.0001, b1 = 0.9, b2 = 0.999 and L2 weight decay of 0.01. In our fine-tuning procedure, we used a linear learning-rate warm-up over the first 2,000 steps. A dropout probability of 0.1 was applied in all layers.

### bert_1760_1900

Refer to the [previous section](#bert) for general description of pre-training procedure for our BERT language models. For training, we used 4x NVIDIA Tesla K80 GPUs, and it took 301 hours for one epoch. We fine-tuned `bert-base-uncased` using the whole dataset with the following hyperparameters:

```python
min_sentence_length=1
model_type=bert
model_name_or_path=bert-base-uncased
mlm_probability=0.15
num_train_epochs=1.0
max_steps=-1
learning_rate=1e-4
weight_decay=0.01 
adam_epsilon=1e-8 
max_grad_norm=1.0 
warmup_steps=2000 
logging_steps=10000 
save_steps=10000 
save_total_limit=10 
per_gpu_train_batch_size=5 
gradient_accumulation_steps=1 
seed=42 
block_size=512 
```

Tokenizer config file:

```python
{"do_lower_case": true, "max_len": 512}
```

`config.json`:

```python
{
  "_num_labels": 2,
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": null,
  "do_sample": false,
  "early_stopping": false,
  "eos_token_ids": null,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "is_decoder": false,
  "is_encoder_decoder": false,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1
  },
  "layer_norm_eps": 1e-12,
  "length_penalty": 1.0,
  "max_length": 20,
  "max_position_embeddings": 512,
  "min_length": 0,
  "model_type": "bert",
  "no_repeat_ngram_size": 0,
  "num_attention_heads": 12,
  "num_beams": 1,
  "num_hidden_layers": 12,
  "num_return_sequences": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pad_token_id": 0,
  "pruned_heads": {},
  "repetition_penalty": 1.0,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 1.0,
  "torchscript": false,
  "type_vocab_size": 2,
  "use_bfloat16": false,
  "vocab_size": 30522
}
```

### bert_1760_1850

We trained this model instance on text published before 1850. The hyperparameters are the same as [bert_1760_1900](#bert_1760_1900). The only differences are: 1) the input text data; 2) fine-tune model: `bert-base-uncased`.

### bert_1850_1875

We trained this model instance on text published between 1850-1875. The hyperparameters are the same as [bert_1760_1900](#bert_1760_1900). The only differences are: 1) the input text data; 2) fine-tune model: `bert_1760_1850`.

### bert_1875_1890

We trained this model instance on text published between 1875-1890. The hyperparameters are the same as [bert_1760_1900](#bert_1760_1900). The only differences are: 1) the input text data; 2) fine-tune model: `bert_1850_1875`.

### bert_1890_1900

We trained this model instance on text published between 1890-1900. The hyperparameters are the same as [bert_1760_1900](#bert_1760_1900). The only differences are: 1) the input text data; 2) fine-tune model: `bert_1875_1890`.