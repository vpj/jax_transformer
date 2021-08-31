# [Autoregressive Transformer Decoder in JAX from scratch](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py)

This implementation builds a transformer decoder from ground up.
This doesn't use any higher level frameworks like Flax and I have used
[labml](https://github.com/labmlai/labml) for logging and experiment tracking.

I have implemented a simple `Module` class to build basic building blocks upon.

This was my first JAX project and many implementations were taken from PyTorch implementations
at [nn.labml.ai](https://nn.labml.ai).

JAX can optimize and differentiate Python pure-functions.
[Pure functions](https://en.wikipedia.org/wiki/Pure_function) are function that take a bunch of
 arguments and return a result without making changes to anything like local variables.
JAX can also compile these functions to as well as vectorize to run them efficiently.

In JAX you don't have to worry about the batches.
The functions are implemented for a single sample and `jax.vit` can vectorize (parallelize) the functions
across the batch dimension (or any other dimension if needed).

### Contents

* [Module class to help us write the layers](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#Module)
* [Embedding layer]https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#Embedding)
* [Positional embeddings](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#PositionalEmbedding)
* [Linear layer](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#Linear)
* [Layer Normalization](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#LayerNormalization)
* [Multi-head attention](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#MHA)
* [Position-wise Feed-Forward layer](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#FFN)
* [TransformerLayer layer](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#TransformerLayer)
* [Cross Entropy Loss](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#CrossEntropyLoss)
* [Autoregressive Transformer](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#AutoregressiveTransformer)
* [Adam Optimizer](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#Adam)
* [Simple dataset](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#Dataset)
* [Experiment code](https://lit.labml.ai/github/vpj/jax_transformer/blob/master/transformer.py#Experiment)

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/1d9b87580a1a11eca4b521ac4d5cf934)
[![Twitter thread](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Flabmlai%2Fstatus%2F1432586070986690560)](https://twitter.com/labmlai/status/1432586070986690560)"""
