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

- [Installation and setup](#installation)
  - [Method 1: Anaconda + install dependencies manually](#method-1)

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
# Install dependencies
```

* To allow the newly created `py38_histLM` environment to show up in the notebooks:

```bash
python -m ipykernel install --user --name py38_histLM --display-name "Python (py38_histLM)"
```
