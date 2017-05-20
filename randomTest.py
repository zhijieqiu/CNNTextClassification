# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:20 2017

@author: zhiq
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100000
x = np.zeros(N,dtype=np.uint64)
x[0] = 1
MAX_INT64 = 18446744073709551615
for i in range(1,N):
    x[i] = (2514903917*x[i-1] +11)% MAX_INT64

x = x/float(MAX_INT64)
y = np.arange(0,1,0.001)