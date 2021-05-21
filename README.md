<div align="center">
    <br>
    <p align="center">
    <h1>histLM</h1>
    </p>
    <h2>Neural network language models for historical research</h2>
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

## Language models

### Download
We have pretrined four types of neural language models using a large historical dataset. These models can be downloaded from [zenodo]().

Currently, we have the following architectures:
- BERT, 5 models
- Flair, 1 model
- word2vec, 2 models
- fastText, 2 models

### Load models

After downloading the language models [here](#download), open the jupyter notebook stored in `codes` directory:

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
