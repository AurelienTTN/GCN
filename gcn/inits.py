#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:58:55 2020

@author: aurelien
"""
import tensorflow as tf
import numpy as np

"L'initialisation fixe le poids des layers"

def uniform(shape,scale=0.05,name=None):
    "On crée un tenseur uniform de dimension shape avec des valeur comprises entre -0.05 et 0.05 max"
    initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
     "Cette intialisation provient d'un papier de Glorot et bengio, les poids sont fixés entre racine carré de 6/(nb entrée poids du layer)"
     init_range = np.sqrt(6.0/(shape[0]+shape[1]))
     initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
     return tf.Variable(initial, name=name)
    
def zeros(shape, name=None):
    "On crée un tenseur de poids fixé à 0"
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    "On crée un tenseur de poids fixé à 1"
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)    


    
    
