#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:07:08 2019

@author: dz
"""

import tensorflow as tf
a = tf.constant([1.0, 2.0], name = "a")
b= tf.constant([2.0, 3.0], name = "b")
result = a + b
print(a.graph is tf.get_default_graph())