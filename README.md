<div align="center">
    <br>
    <p align="center">
    <h1>histLM</h1>
    </p>
    <h2>Neural language models for historical research</h2>
</div>
 
<p align="center">
    <a href="./LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <br/>
</p>

Table of contents
-----------------

- [Language models](#language-models)
    - [Download](#architectures)
    - [Load models](#load-models)
- [Installation and setup](#installation)
  - [Method 1: Anaconda + install dependencies manually](#method-1)
- [Language models in use](#language-models-in-use)

## Language models

### Download
We have pretrined four types of neural language models using a large historical dataset. These models can be downloaded from [zenodo](). The directory structure is as follows:

```bash
histLM_dataset
├── README.md
├── bert
│   ├── bert_1760_1850
│   ├── bert_1760_1900
│   ├── bert_1850_1875
│   ├── bert_1875_1890
│   └── bert_1890_1900
├── fasttext
│   ├── ft_1760_1850
│   └── ft_1760_1900
├── flair
│   └── flair_1760_1900
└── word2vec
    ├── w2v_1760_1850
    └── w2v_1760_1900
```

Currently, we have the following architectures:
- BERT, 5 models
- Flair, 1 model
- word2vec, 2 models
- fastText, 2 models

### Load models

After downloading the language models (refer to [Download section](#download)), put the uncompressed directory inside `histLM` directory:

```bash
histLM
├── README.md
├── codes
│   └── *.py files
└── histLM_dataset
    ├── README.md
    ├── bert
    │   ├── bert_1760_1850
    │   ├── bert_1760_1900
    │   ├── bert_1850_1875
    │   ├── bert_1875_1890
    │   └── bert_1890_1900
    ├── fasttext
    │   ├── ft_1760_1850
    │   └── ft_1760_1900
    ├── flair
    │   └── flair_1760_1900
    └── word2vec
        ├── w2v_1760_1850
        └── w2v_1760_1900
```

Next, open one of the jupyter notebooks stored in `codes` directory:

```bash
$ cd codes
$ jupyter notebook
```

## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

### Method 1

* Create a new environment for `histLM` called `py38_histLM`:

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
pip install notebook
```

* To allow the newly created `py38_histLM` environment to show up in the notebooks:

```bash
python -m ipykernel install --user --name py38_histLM --display-name "Python (py38_histLM)"
```

## Language models in use

So far, the language models presented in this repository have been used in the following projects:
* Living Machines: A Study of Atypical Animacy (COLING 2020): [repository](https://github.com/Living-with-machines/AtypicalAnimacy) and [paper](https://www.aclweb.org/anthology/2020.coling-main.400/).
* When Time Makes Sense: A Historically-Aware Approach to Targeted Sense Disambiguation (Findings of ACL 2021): [repository](https://github.com/Living-with-machines/TargetedSenseDisambiguation) and paper (forthcoming).
* Assessing the Impact of OCR Quality on Downstream NLP Tasks (ARTIDIGH 2020): [repository](https://github.com/Living-with-machines/lwm_ARTIDIGH_2020_OCR_impact_downstream_NLP_tasks) and [paper](https://www.repository.cam.ac.uk/handle/1810/304987).
