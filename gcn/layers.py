#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###
#Created on Wed Jul 29 10:07:22 2020

#@author: aurelien
###

from inits import *
import tensorflow as tf


# Dictionnaire des id de nos layers
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    #Cette fonction permet d'assigner un nom unique à chaque layer#
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    #Le dropout est utilisé afin de ne pas surentrainé notre réseau, on va perturber nous même l'apprentissage en fixant un pourcentage de poids à 0#
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    #cette fonction permet de multiplié deux matrices en utilisant la méthode la plus approprié suivant leur forme#
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res
    

class Layer(object):

    #Un layer possède un nom qui définit la porté du layer, le logging permet de passer la représentation de notre layer sous forme d'histogramme sur 0/1.#
  
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    #La fonction call prend les entrées du layer et retourne les sorties#        
    def _call(self, inputs):
        return inputs
    
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs    

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
            
class Dense(Layer):
    #Ici on déifnit un layer dense, #
    
    
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        self.num_features_nonzero = placeholders['num_features_nonzero']

        #on fixe les poids du layer grâce à l'initilisation de glorot et s'il y a des biais on les fixe sur zeros#
	
	
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
	
	   #Si l'affichage est sur on, on affiche toutes les données du layer#
        if self.logging:
            self._log_vars()
            
	
    def _call(self, inputs):
    
    	#On prend x en valeur d'entrée
        x = inputs

        # On applique le dropout 
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # On applique la pondération aux entrée du layer
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # On ajoute les biais 
        if self.bias:
            output += self.vars['bias']
	
	# Enfin on retourne le tout
        return self.act(output)
        
        
class GraphConvolution(Layer):
    ###On définit la fonction pour créer un layer convolutionnel###
    
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        
        #On prend les inputs X
        x = inputs

        # On applique un dropout sur X
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # On ajoute les biais s'il y en a avant de retourner l'output qui passe par une fonction relu 
        if self.bias:
            output += self.vars['bias']
            
        return self.act(output)
        
if __name__=='__main__':
    pass


