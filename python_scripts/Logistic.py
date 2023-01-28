# -*- coding: utf-8 -*-
"""
Logstic Function / Sigmoid Function
@author: TSE
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

inputNum = np.arange(-5.0,5.0,0.1)

y_sigmoid = sigmoid(inputNum)


plt.plot(y_sigmoid,marker="o")
plt.show()












