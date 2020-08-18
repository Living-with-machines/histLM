# BERT: train a new LM from scratch or Fine-Tuning an existing LM (Feb. 2020)

## First successful run

**Dataset: books published `<= 1850`**

- Preprocess text data (all books):

```bash
cd /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020
python step02_preproc_no_ngrams_file_output_v003.py
```

- Combine all books together and create one large file (otherwise, tokenizer will complain):

```bash
cd codes
python combine_books.py
```

This creates a huge file with 208239904 lines called `all_books_v003.txt`.

- Train a tokenizer (NOT NEEDED if the goal is to fine-tune the original BERT LM)

I experimented with three types of tokenizers (refer to the last one):

```python
from tokenizers import BertWordPieceTokenizer
bert_tokenizer = BertWordPieceTokenizer()
bert_tokenizer.train(files="./datasets/preprocessing_Feb25_2020/outputs/all_books_v003.txt", vocab_size=300000, min_frequency=100, limit_alphabet=5000)
bert_tokenizer.save(".", "book_DB_300K_100_5K")
```

This step is not really needed, but I also did:

```python
from transformers import tokenization_bert
mytok_bert = tokenization_bert.BertTokenizer("./book_DB_300K_100_5K-vocab.txt")
mytok_bert.save_pretrained(".")
```

and moved all the outputs to:

```bash
/datadrive/khosseini/LM_with_bert/datasets/vocabs/BertWordPieceTokenizer_book_DB_300K_100_5K
```

To check the parameters in the original model:

```python
from transformers import BertTokenizer
bert_original = BertTokenizer.from_pretrained("bert-base-uncased")
bert_original.vocab_size
bert_original.special_tokens_map
```

Another type of tokenizer that is suggested by Huggingface is `ByteLevelBPETokenizer`. To train this:

```python
from tokenizers import ByteLevelBPETokenizer
bert_tokenizer = ByteLevelBPETokenizer()
bert_tokenizer.train(files="./datasets/preprocessing_Feb25_2020/outputs/all_books_v003.txt", vocab_size=300000, min_frequency=100)

# ATTENTION: you probably want to also add:

bert_tokenizer.train(files="./datasets/preprocessing_Feb25_2020/outputs/all_books_v003.txt", vocab_size=300000, min_frequency=100, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
```

And moved all the outputs here:

```bash
/datadrive/khosseini/LM_with_bert/datasets/vocabs/ByteLevelBPETokenizer_book_DB_300K_100
```

Unfortunately, `book_DB_300K_100_5K` took lots of space and took a very long time to train, so I changed the tokenizer as follows:

```python
from tokenizers import BertWordPieceTokenizer
bert_tokenizer = BertWordPieceTokenizer()
bert_tokenizer.train(files="./datasets/preprocessing_Feb25_2020/outputs/all_books_v003.txt", vocab_size=100000, min_frequency=200, limit_alphabet=5000)
bert_tokenizer.save(".", "book_DB_100K_200_5K")
```

- Finally:

### Fine-tune a LM

For Fine-Tuning a language-model, first, we need to create an evaluation set, for that:

```bash
cd /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003
# open file /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003/create_evaluation_set.py
# change the parameters
# This code moves some files to /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003/evaluation_v001
python /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003/create_evaluation_set.py
```

In the next step, we create one file out of evaluation set:

```bash
cd /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003
python /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003/combine_evaluation_books.py
```

For Fine-tuning, we use the original tokenizer (note that `--tokenizer_name` is not set):

```python
# warmup_steps=2000 was selected based on the original BERT paper in which they used 10% of the total number of steps (~1M) to warmup

python ./codes/transformers/examples/run_language_modeling_v001.py \
       --min_sentence_length=1 \
       --max_date=1850 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003/words_*.txt" \
       --output_dir=FT_bert_base_uncased_before_1850_v001 \
       --cached_lm_suffix=FT_bert_base_uncased_before_1850_v001 \
       --model_type=bert \
       --model_name_or_path=bert-base-uncased \
       --mlm \
       --mlm_probability=0.15 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --block_size=512 \

# NOT USED:
       --do_eval \
       --evaluate_during_training \
       --eval_data_file=/datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003/evaluation_v001/evaluation_v001.txt \
       --per_gpu_eval_batch_size=5
```

After creating the model for dates <= 1850 stored here: `./models/bert/FT_bert_base_uncased_before_1850_v001/`. Next, we FT the already created LM on newer books. In order to have the same number of characters in the new batch:

```python
# you need to run this only one, this create a DataFrame which contains the following information for each book:
# ["path", "length", "language", "date", "genre", "type"]
python /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/create_books_df.py
```

This script generates: `/datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/all_books.pkl`. Next:

```python
python /datadrive/khosseini/LM_with_bert/datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/check_length.py
```

When the dates are identified, we start the fine-tuning by:

```python
# warmup_steps=2000 was selected based on the original BERT paper in which they used 10% of the total number of steps (~1M) to warmup

python ./codes/transformers/examples/run_language_modeling_v002.py \
       --min_sentence_length=1 \
       --min_date=1850 \
       --max_date=1875 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/all_books.pkl" \
       --output_dir=FT_bert_base_uncased_after_1850_before_1875_v002 \
       --cached_lm_suffix=FT_bert_base_uncased_after_1850_before_1875_v002 \
       --model_type=bert \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_before_1850_v001" \
       --mlm \
       --mlm_probability=0.15 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --block_size=512 \
```

Unfortunately, I had to kill that job after 11K iterations!! To restart from the last checkpoint:

```python
python ./codes/transformers/examples/run_language_modeling_v002.py \
       --min_sentence_length=1 \
       --min_date=1850 \
       --max_date=1875 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/all_books.pkl" \
       --output_dir=FT_bert_base_uncased_after_1850_before_1875_v002 \
       --cached_lm_suffix=FT_bert_base_uncased_after_1850_before_1875_v002 \
       --model_type=bert \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_before_1850_v001" \
       --mlm \
       --mlm_probability=0.15 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --block_size=512 \
       --should_continue \
       --overwrite_output_dir
```

Now for 1875-1890:

```python
# warmup_steps=2000 was selected based on the original BERT paper in which they used 10% of the total number of steps (~1M) to warmup

python ./codes/transformers/examples/run_language_modeling_v002.py \
       --min_sentence_length=1 \
       --min_date=1875 \
       --max_date=1890 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/all_books.pkl" \
       --output_dir=FT_bert_base_uncased_after_1875_before_1890_v002 \
       --cached_lm_suffix=FT_bert_base_uncased_after_1875_before_1890_v002 \
       --model_type=bert \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_after_1850_before_1875_v002" \
       --mlm \
       --mlm_probability=0.15 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --block_size=512 \
```

for 1890-1900:

```python
# warmup_steps=2000 was selected based on the original BERT paper in which they used 10% of the total number of steps (~1M) to warmup

python ./codes/transformers/examples/run_language_modeling_v002.py \
       --min_sentence_length=1 \
       --min_date=1890 \
       --max_date=1900 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/all_books.pkl" \
       --output_dir=FT_bert_base_uncased_after_1890_before_1900_v002 \
       --cached_lm_suffix=FT_bert_base_uncased_after_1890_before_1900_v002 \
       --model_type=bert \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_after_1875_before_1890_v002" \
       --mlm \
       --mlm_probability=0.15 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --block_size=512 \
```

In this step, the goal is to fine-tune language models using human-related sentences, for this, I changed the `run_language_modeling.py` (see `/datadrive/khosseini/LM_with_bert/codes/transformers/examples/run_language_modeling_v004.py`).

```bash
python ./codes/transformers/examples/run_language_modeling_v004.py \
       --min_sentence_length=1 \
       --do_train \
       --train_data_file="./datasets/selected_sentences/human_related/pre1850/*.tsv" \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_before_1850_v001" \
       --output_dir=FT_bert_pre1850_with_human_sents \
       --model_type=bert \
       --mlm \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=500 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --line_by_line_dataframe \
       --block_size=512 \
```

```bash
python ./codes/transformers/examples/run_language_modeling_v004.py \
       --min_sentence_length=1 \
       --do_train \
       --train_data_file="./datasets/selected_sentences/human_related/1850to1875/*.tsv" \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_after_1850_before_1875_v002/" \
       --output_dir=FT_bert_1850to1875_with_human_sents \
       --model_type=bert \
       --mlm \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=500 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --line_by_line_dataframe \
       --block_size=512 \
```

```bash
python ./codes/transformers/examples/run_language_modeling_v004.py \
       --min_sentence_length=1 \
       --do_train \
       --train_data_file="./datasets/selected_sentences/human_related/1875to1890/*.tsv" \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_after_1875_before_1890_v002/" \
       --output_dir=FT_bert_1875to1890_with_human_sents \
       --model_type=bert \
       --mlm \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=500 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --line_by_line_dataframe \
       --block_size=512 \
```

```bash
python ./codes/transformers/examples/run_language_modeling_v004.py \
       --min_sentence_length=1 \
       --do_train \
       --train_data_file="./datasets/selected_sentences/human_related/1890to1900/*.tsv" \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_after_1890_before_1900_v002/" \
       --output_dir=FT_bert_1890to1900_with_human_sents \
       --model_type=bert \
       --mlm \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=500 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --line_by_line_dataframe \
       --block_size=512 \
```








### VERY IMPORTANT

Some questions:

- Is it possible to have more than two [SEP] in one training sequence (I don’t think it is possible, but needs to be checked)
- How long would it take to train the model using --line-by-line option (the last time that I checked it took around 20 days for one time window, but needs to be checked). If it takes a very long time, test it for one time window.
- How many [MASK] tokens we can have in one training sequence

The way that we are adding-special-characters is a a bit weird! We are now using:

```python
tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))

# Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
# If your dataset is small, first you should loook for a bigger one :-) and second you
# can change this behavior by adding (model specific) padding.
```

The problem here is that we do not add [SEP] or [CLS] for all sentences, but rather to 510 tokens (which can consist of many sentences!!!). I think, this needs to be changed as follows:

```python
tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text.replace("\n", "[SEP] ")))
```

Also test the `--line_by_line` option! That should be also interesting, but most importantly!!!

TODO: follow the Esperanto example. How do they add these special tokens?

This can be tested by:

```bash
# TESTING!!!
python ./codes/transformers/examples/run_language_modeling_v004.py \
       --min_sentence_length=1 \
       --min_date=1913 \
       --max_date=1914 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003_all/others/all_books.pkl" \
       --output_dir=TEST_BERT \
       --cached_lm_suffix=TEST_BERT \
       --model_type=bert \
       --model_name_or_path="./models/bert/FT_bert_base_uncased_after_1850_before_1875_v002" \
       --mlm \
       --mlm_probability=0.15 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=2000 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit=10 \
       --per_gpu_train_batch_size=5 \
       --gradient_accumulation_steps=1 \
       --seed=42 \
       --line_by_line \
       --block_size=512
```










```bash
python ./codes/transformers/examples/run_language_modeling_v001.py \
       --min_sentence_length=1 \
       --max_date=1850 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003/words_*.txt" \
       --output_dir=FT_bert_base_uncased_before_1850 \
       --cached_lm_suffix=FT_bert_base_uncased_before_1850 \
       --model_type=bert \
       --model_name_or_path=bert-base-uncased \
       --tokenizer_name="./datasets/vocabs/BertWordPieceTokenizer_book_DB_100K_200_5K/vocab.txt" \
       --mlm \
       --mlm_probability=0.2 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=5e-5 \
       --weight_decay=0.0 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=0 \
       --logging_steps=10000 \
       --save_steps=10000 \
       --save_total_limit 10 \
       --per_gpu_train_batch_size=3 \
       --gradient_accumulation_steps=1 \
       --seed 42 \
       --block_size=512


# Not used options:
--line_by_line \
--should_continue

--do_eval
--eval_data_file
--evaluate_during_training
--eval_all_checkpoints
--per_gpu_eval_batch_size

--no_cuda
--save_total_limit
--overwrite_output_dir
--overwrite_cache
--cache_dir
--config_name

--fp16
--fp16_opt_level

--local_rank
--server_ip
--server_port
```




```bash
./models/bert/FT_bert_base_uncased_before_1850_line_by_line_2_percent

python ./codes/transformers/examples/run_language_modeling_v001.py \
       --min_sentence_length=5 \
       --max_date=1850 \
       --do_train \
       --train_data_file="./datasets/preprocessing_Feb25_2020/outputs/preproc_v003/words_*.txt" \
       --output_dir=FT_bert_base_uncased_before_1850 \
       --cached_lm_suffix=FT_bert_base_uncased_before_1850 \
       --model_type=bert \
       --model_name_or_path=bert-base-uncased \
       --tokenizer_name="./datasets/vocabs/BertWordPieceTokenizer_book_DB_100K_200_5K/vocab.txt" \
       --line_by_line \
       --mlm \
       --mlm_probability=0.2 \
       --num_train_epochs=1.0 \
       --max_steps=-1 \
       --learning_rate=5e-5 \
       --weight_decay=0.0 \
       --adam_epsilon=1e-8 \
       --max_grad_norm=1.0 \
       --warmup_steps=0 \
       --logging_steps=5000 \
       --save_steps=5000 \
       --save_total_limit 10 \
       --per_gpu_train_batch_size=2 \
       --gradient_accumulation_steps=1 \
       --seed 42 \
       --block_size=512 \


# Not used options:
--should_continue

--do_eval
--eval_data_file
--evaluate_during_training
--eval_all_checkpoints
--per_gpu_eval_batch_size

--no_cuda
--save_total_limit
--overwrite_output_dir
--overwrite_cache
--cache_dir
--config_name

--fp16
--fp16_opt_level

--local_rank
--server_ip
--server_port
```



## Other experiments

First, I tried the following command. However, I had to kill this job since it would take around 440 hours on 2 GPUs:

```bash
$ python ./transformers/examples/run_language_modeling.py --output_dir=learn_fom_scratch_all_books --model_type=bert --do_train --train_data_file="/datadrive/fnanni/codes/Living-with-Machines-code/language_models/notebooks/steps2lang_model/books/scripts/exp_002/outputs/corpus/train/words_*.txt" --mlm --tokenizer_name bert-base-uncased
```

**NOTE** you need to move all `bert_cache`... stored in the .../outputs/corpus/train/ directory.

I also tried to run the following command, but it failed. My guess is that there is something wrong with `Vocab_all_min_count_5.txt`. This issue is now solved. See below.

```bash
python ./transformers/examples/run_language_modeling.py --output_dir=learn_fom_scratch_all_books --model_type=bert --do_train --train_data_file=/datadrive/fnanni/codes/Living-with-Machines-code/language_models/notebooks/steps2lang_model/books/scripts/exp_002/outputs/corpus/all_books.txt --mlm --tokenizer_name bert-vocab-builder/Vocab_all_min_count_5.txt
```

The `Vocab_all_min_count_5` was generated using:

```bash
python subword_builder.py --corpus_filepattern "/datadrive/fnanni/codes/Living-with-Machines-code/language_models/notebooks/steps2lang_model/books/scripts/exp_002/outputs/corpus/train/words_*.txt" --output_filename "Vocab_all_min_count_5.txt" --min_count 5
```

Next, I changed the file (`run_language_modeling`) to only consider books that were published <= 1850 and that are in English ("en" in the header)

```bash
python ./transformers/examples/run_language_modeling.py --output_dir=FT_bert_base_uncased_before_1850 --model_type=bert --model_name_or_path=bert-base-uncased --do_train --train_data_file="/datadrive/fnanni/codes/Living-with-Machines-code/language_models/notebooks/steps2lang_model/books/scripts/exp_002/outputs/corpus/train/words_*.txt" --mlm --tokenizer_name bert-base-uncased
```

This was also killed! the main reason is that we are saving checkpoints too frequently. So in the next step, I made several changes:


### MISC

# ---------------------

python ./transformers/examples/run_language_modeling.py \
       --do_train \
       --train_data_file="/datadrive/fnanni/codes/Living-with-Machines-code/language_models/notebooks/steps2lang_model/books/scripts/exp_002/outputs/corpus/train/words_*.txt" \
       --output_dir=FT_bert_base_uncased_before_1850 \
       --model_type=bert \
       --model_name_or_path=bert-base-uncased \          # The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.
       --tokenizer_name bert-base-uncased \             # Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.
       --line_by_line \
       --mlm \
       --mlm_probability 0.2 \
       --num_train_epochs 1.0 \
       --max_steps -1 \                                  # If > 0: set total number of training steps to perform. Override num_train_epochs.
       --learning_rate 5e-5 \                            # The initial learning rate for Adam.
       --weight_decay 0.0 \                              # Weight decay if we apply some.
       --adam_epsilon 1e-8 \                             # Epsilon for Adam optimizer.
       --max_grad_norm 1.0 \                             # Max gradient norm.
       --warmup_steps 0 \                                # Linear warmup over warmup_steps.
       --logging_steps 20000 \
       --save_steps 20000 \
       --block_size -1 \                                 # Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training.
       --per_gpu_train_batch_size 4 \                    # Batch size per GPU/CPU for training.
       --gradient_accumulation_steps 1 \                 # Number of updates steps to accumulate before performing a backward/update pass.

       #--should_continue                                # Whether to continue from latest checkpoint in output_dir
       #--do_eval
       #--eval_data_file
       #--evaluate_during_training                       # Run evaluation during training at each logging step.
       #--eval_all_checkpoints
       #--per_gpu_eval_batch_size                        # Batch size per GPU/CPU for evaluation.
       #--save_total_limit
       #--no_cuda
       #--overwrite_output_dir
       #--overwrite_cache
       #--cache_dir
       #--seed
       #--fp16
       #--fp16_opt_level
       #--local_rank
       #--server_ip
       #--server_port
       #--config_name

# ---------------------

# Arguments for run_language_model

  -h, --help            show this help message and exit
  --train_data_file TRAIN_DATA_FILE
                        The input training data file (a text file).
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --model_type MODEL_TYPE
                        The model architecture to be trained or fine-tuned.
  --eval_data_file EVAL_DATA_FILE
                        An optional input evaluation data file to evaluate the
                        perplexity on (a text file).
  --line_by_line        Whether distinct lines of text in the dataset are to
                        be handled as distinct sequences.
  --should_continue     Whether to continue from latest checkpoint in
                        output_dir
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization. Leave
                        None if you want to train a model from scratch.
  --mlm                 Train with masked-language modeling loss instead of
                        language modeling.
  --mlm_probability MLM_PROBABILITY
                        Ratio of tokens to mask for masked language modeling
                        loss
  --config_name CONFIG_NAME
                        Optional pretrained config name or path if not the
                        same as model_name_or_path. If both are None,
                        initialize a new config.
  --tokenizer_name TOKENIZER_NAME
                        Optional pretrained tokenizer name or path if not the
                        same as model_name_or_path. If both are None,
                        initialize a new tokenizer.
  --cache_dir CACHE_DIR
                        Optional directory to store the pre-trained models
                        downloaded from s3 (instead of the default one)
  --block_size BLOCK_SIZE
                        Optional input sequence length after tokenization.The
                        training dataset will be truncated in block of this
                        size for training.Default to the model max input
                        length for single sentence inputs (take into account
                        special tokens).
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --evaluate_during_training
                        Run evaluation during training at each logging step.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints, delete the
                        older checkpoints in the output_dir, does not delete
                        by default
  --eval_all_checkpoints
                        Evaluate all checkpoints starting with the same prefix
                        as model_name_or_path ending and ending with step
                        number
  --no_cuda             Avoid using CUDA when available
  --overwrite_output_dir
                        Overwrite the content of the output directory
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --seed SEED           random seed for initialization
  --fp16                Whether to use 16-bit (mixed) precision (through
                        NVIDIA apex) instead of 32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --server_ip SERVER_IP
                        For distant debugging.
  --server_port SERVER_PORT
                        For distant debugging.
