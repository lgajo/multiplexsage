import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros

class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu,
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['intra_neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='intra_neigh_weights')
            self.vars['inter_neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='inter_neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs, slicer):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        intra_neigh_vecs = tf.slice(neigh_vecs, [0,0,0], [-1, slicer, -1])
        inter_neigh_vecs = tf.slice(neigh_vecs, [0,slicer,0], [-1, -1, -1])
        intra_neigh_means = tf.reduce_mean(intra_neigh_vecs, axis=1)
        inter_neigh_means = tf.reduce_mean(inter_neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_intra_neighs = tf.matmul(intra_neigh_means, self.vars['intra_neigh_weights'])
        from_inter_neighs = tf.matmul(inter_neigh_means, self.vars['inter_neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_intra_neighs, from_inter_neighs])
        else:
            output = tf.concat([from_self, from_intra_neighs, from_inter_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def __call__(self, inputs,slicer):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs,slicer)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs
