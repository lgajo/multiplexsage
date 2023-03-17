## MultiplexSAGE: a multiplex embedding algorithm

#### Authors: Luca Gallo (gallol@ceu.edu), Vito Latora, Alfredo Pulvirenti

### Overview

This directory contains code for the MultiplexSAGE algorithm, a generalization of the GraphSAGE algorithm that allows embedding multiplex networks.
For more details on the algorithm, check out our [preprint](https://arxiv.org/abs/2206.13223.pdf).

The Graphs directory contains some examples analyzed in the paper. We provide three example graphs, one in each subdirectory: 
1) arxiv: it is a collaboration network where each layer represents a different category of the pre-print archive. [Source](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.011027)
2) drosophila: it is the protein-genetic interaction network of the common fruit fly Drosophila melanogaster Each layer corresponds to a different type of interaction. [Source](https://www.nature.com/articles/ncomms7864)
3) fftwyt: it is a multiplex network obtained from Friendfeed (ff), a social media aggregator, which allows the users to register their accounts on other online social networks. [Source](https://www.cambridge.org/core/books/multilayer-social-networks/39383306D9843313057CECEBF7B9BF26)

If you make use of this code or the MultiplexSAGE algorithm in your work, please cite the following paper:

     @article{gallo2022multiplexsage,
              title={MultiplexSAGE: a multiplex embedding algorithm for inter-layer link prediction},
              author={Gallo, Luca and Latora, Vito and Pulvirenti, Alfredo},
              journal={arXiv preprint arXiv:2206.13223},
               year={2022}
}

### How to run

You can run MultiplexSage inside a docker image:

     $ docker build -t multiplexsage .
     $ docker run -it multiplexsage bash

The example_multiplexsage.sh contains example usages of the model. In this example, the algorithm task is intra-layer and inter-layer link prediction, which is an unsupervised task. 

### Model 
The user must specify a --model flag. Currently, only the mean aggregator is provided, but we plan to extend the code to other aggregating functions.

### Data
Two input files have to be provided to the model:

1) multi-G.json -- A networkx-specified json file describing the input graph. Edges have an attribute to categorize them as intra-layer and inter-layer, as well as an attribute to define whether they are in the training or in the test set.
2) multi-id_map.json -- A dictionary mapping the multiplex ids of the nodes to consecutive integers. Note that same nodes in different layers have different ids.

The flags --n_layers specifies the number of layers in the multiplex network, while --n_try flag can be used to select a specific graph if multiple train-test splits of the same network are considered.

### Acknowledgements

This code is based on a fork of the [GraphSAGE code](https://github.com/williamleif/GraphSAGE), which is described in this [paper](https://arxiv.org/abs/1706.02216).
