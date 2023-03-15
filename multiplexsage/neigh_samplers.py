from __future__ import division
from __future__ import print_function

from multiplexsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, intra_adj_info, inter_adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.intra_adj_info = intra_adj_info
        self.inter_adj_info = inter_adj_info

    def _call(self, inputs):
        ids, num_samples, num_sheets = inputs
        adj_lists = tf.nn.embedding_lookup(self.intra_adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        inter_adj_lists = tf.nn.embedding_lookup(self.inter_adj_info, ids)
        inter_adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(inter_adj_lists)))
        inter_adj_lists = tf.slice(inter_adj_lists, [0,0], [-1, num_sheets])
        adj_lists = tf.concat([adj_lists, inter_adj_lists],1)
        return adj_lists
