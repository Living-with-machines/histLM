<div align="center">
    <br>
    <p align="center">
    <h1>histLM</h1>
    </p>
    <h2>Neural Language Models for Historical Research</h2>
</div>
 
<p align="center">
    <a href="https://github.com/Living-with-machines/histLM/workflows/Continuous%20integration/badge.svg">
        <img alt="Continuous integration badge" src="https://github.com/Living-with-machines/histLM/workflows/Continuous%20integration/badge.svg">
    </a>
    <br/>
</p>

Table of contents
-----------------

- [Language models](#language-models)
    - [Download from zenodo](#download-from-zenodo)
    - [Download BERT models from Hugging Face](#download-bert-models-from-hugging-face)
    - [Load models](#load-models)
- [Language models in use](#language-models-in-use)
- [Installation and setup](#installation)
  - [Anaconda + install dependencies manually](#installation)
- [How to cite](#how-to-cite)
- [License](#license)

## Language models

### Download from zenodo
We have pre-trained four types of neural language models trained on a large historical dataset of books in English, published between 1760-1900 and comprised of ~5.1 billion tokens. The language model architectures include word type embeddings (**word2vec** and **fastText**) and contextualized models (**BERT** and **Flair**). For each architecture, we trained a model instance using the whole dataset. Additionally, we trained separate instances on text published before 1850 for the type embeddings (i.e., **word2vec** and **fastText**), and four instances considering different time slices for BERT.

:warning: The language models can be downloaded from [zenodo](http://doi.org/10.5281/zenodo.4782245). (see [License](#license))

Each `.zip` file on [zenodo](http://doi.org/10.5281/zenodo.4782245) contains model instances for one neural network architecture (i.e., bert, flair, fasttext and word2vec). After unzipping the four .zip files, the directory structure is as follows:

```bash=
histLM_dataset
├── README.md
├── bert
│   ├── bert_1760_1850
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.txt
│   ├── bert_1760_1900
│   |   └── ...
│   ├── bert_1850_1875
│   |   └── ...
│   ├── bert_1875_1890
│   |   └── ...
│   └── bert_1890_1900
│       └── ...
|
├── flair
│   └── flair_1760_1900
│       ├── best-lm.pt
│       ├── loss.txt
│       └── training.log
|
├── fasttext
│   ├── ft_1760_1850
│   │   ├── fasttext_words.model
│   │   ├── fasttext_words.model.trainables.syn1neg.npy
│   │   ├── fasttext_words.model.trainables.vectors_ngrams_lockf.npy
│   │   ├── fasttext_words.model.trainables.vectors_vocab_lockf.npy
│   │   ├── fasttext_words.model.wv.vectors.npy
│   │   ├── fasttext_words.model.wv.vectors_ngrams.npy
│   │   └── fasttext_words.model.wv.vectors_vocab.npy
│   └── ft_1760_1900
│       └── ...
|
└── word2vec
    ├── w2v_1760_1850
    │   ├── w2v_words.model
    │   ├── w2v_words.model.trainables.syn1neg.npy
    │   └── w2v_words.model.wv.vectors.npy
    └── w2v_1760_1900
        └── ...
```

### Download BERT models from Hugging Face

In addition to downloading the [models from zenodo](#download-from-zenodo), the BERT models can be downloaded from Hugging Face Hub, see: https://huggingface.co/Livingwithmachines

- bert_1760_1900 : https://huggingface.co/Livingwithmachines/bert_1760_1900
- bert_1760_1850: https://huggingface.co/Livingwithmachines/bert_1760_1850
- bert_1850_1875: https://huggingface.co/Livingwithmachines/bert_1850_1875
- bert_1875_1890: https://huggingface.co/Livingwithmachines/bert_1875_1890
- bert_1890_1900: https://huggingface.co/Livingwithmachines/bert_1890_1900

### Load models

After downloading the language models from [zenodo](http://doi.org/10.5281/zenodo.4782245) (refer to [Download section](#download)):

1. Go to `histLM` directory:

```bash
cd /path/to/histLM
```

2. Create a directory called `histLM_dataset`:

```bash
mkdir histLM_dataset
```

3. Move the unzipped directories to `histLM/histLM_dataset`. The directory structure should be:

```bash
histLM
├── LICENSE
├── README.md
├── histLM_dataset
│   ├── README.md
│   ├── bert
│   │   ├── bert_1760_1900
│   │   ├── bert_1760_1850
│   │   ├── bert_1850_1875
│   │   ├── bert_1875_1890
│   │   └── bert_1890_1900
│   ├── fasttext
│   │   ├── ft_1760_1850
│   │   └── ft_1760_1900
│   ├── flair
│   │   └── flair_1760_1900
│   └── word2vec
│       ├── w2v_1760_1850
│       └── w2v_1760_1900
├── notebooks
│   ├── BERT_model.ipynb
│   ├── Flair_model.ipynb
│   ├── fastText_model.ipynb
│   └── word2vec_model.ipynb
├── requirements.txt
└── tests
    └── test_import.py
```

4. Finally, open one of the jupyter notebooks stored in the `notebooks` directory:

```bash
$ cd notebooks
$ jupyter notebook
```

## Language models in use

So far, the language models presented in this repository have been used in the following projects:
* When Time Makes Sense: A Historically-Aware Approach to Targeted Sense Disambiguation (Findings of ACL: ACL-IJCNLP 2021): [repository](https://github.com/Living-with-machines/TargetedSenseDisambiguation) and [paper](https://aclanthology.org/2021.findings-acl.243/).
* Living Machines: A Study of Atypical Animacy (COLING 2020): [repository](https://github.com/Living-with-machines/AtypicalAnimacy) and [paper](https://www.aclweb.org/anthology/2020.coling-main.400/).
* Assessing the Impact of OCR Quality on Downstream NLP Tasks (ARTIDIGH 2020): [repository](https://github.com/Living-with-machines/lwm_ARTIDIGH_2020_OCR_impact_downstream_NLP_tasks) and [paper](https://www.repository.cam.ac.uk/handle/1810/304987).
* 'The Living Machine: A Computational Approach to the Nineteenth-Century Language of Technology" in _Technology and Culture_ (2023) [Paper](https://muse.jhu.edu/pub/1/article/903976) and [Repository](https://github.com/Living-with-machines/TheLivingMachine)

## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

* After installing Anaconda, create a new environment for `histLM` called `py38_histLM`:

```bash
conda create -n py38_histLM python=3.8
```

* Activate the environment:

```bash
conda activate py38_histLM
```

* Clone `histLM` source code:

```bash
git clone https://github.com/Living-with-machines/histLM.git 
```

* Install dependencies:

```
pip install -r requirements.txt
```

Alternatively:

```
pip install torch==1.9.0
pip install transformers==4.10.0
pip install flair==0.9
pip install gensim==3.8.3
pip install notebook==6.4.3
pip install jupyter-client==7.0.2
pip install jupyter-core==4.7.1
pip install ipywidgets==7.6.4
```

* To allow the newly created `py38_histLM` environment to show up in the notebooks:

```bash
python -m ipykernel install --user --name py38_histLM --display-name "Python (py38_histLM)"
```

## How to cite

To cite histLM or any of the language models: 

```text
Hosseini, K., Beelen, K., Colavizza, G., & Coll Ardanuy, M. (2021). Neural Language Models for Nineteenth-Century English. Journal of Open Humanities Data, 7: 22, pp. 1–6. DOI: https://doi.org/10.5334/johd.48
```

Link (Journal of Open Humanities Data): http://doi.org/10.5334/johd.48

## License

**Codes/notebooks** are released under MIT License.

**Models** are released under open license CC BY 4.0, available at https://creativecommons.org/licenses/by/4.0/legalcode.
