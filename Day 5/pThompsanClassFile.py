# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:04:39 2022

@author: TSE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
simulation_dataset = pd.read_csv("Ads ClickThroughRate.csv")

d = 10
N = 10000

number_of_times_reward_0 = [0]*d
number_of_times_reward_1 = [0]*d
ad_displayed = []

for n in range(0,N):
    max_dist=0
    for i in range(0,d):
        
        current_dist = random.betavariate(number_of_times_reward_1[i]+1, number_of_times_reward_0[i]+1)
        
        if current_dist > max_dist:
            max_dist = current_dist
            ad = i

    ad_displayed.append(ad)
    
    clicked_reward = simulation_dataset.values[n,ad]
    
    if clicked_reward==1:
        number_of_times_reward_1[ad] += 1
    else:
        number_of_times_reward_0[ad] += 1


plt.hist(ad_displayed)
plt.show()






