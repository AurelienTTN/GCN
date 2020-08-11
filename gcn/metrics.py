#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    #On applique ici un softmax cross entropy au prediction par les labels
    #Cela revient à mesurer la probabilité d'erreur sur des valeurs discrètes
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    #On calcule l'accuracy de notre modèle
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)