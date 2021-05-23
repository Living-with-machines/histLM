import pytest

def test_import():
    from transformers import pipeline
    
    from flair.embeddings import FlairEmbeddings
    from flair.data import Sentence

    import flair, torch

    from gensim.models import FastText

    from gensim.models import Word2Vec


