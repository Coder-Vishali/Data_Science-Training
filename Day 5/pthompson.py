"""
Thompson Sampling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
simulation_dataset = pd.read_csv("Ads ClickThroughRate.csv")

d = 10
N = 10000

number_of_times_reward_1 = [0]*d
number_of_times_reward_0 = [0]*d
ad_display =[]

for n in range(0,N):
    max_dist=0
    ad=0
    for i in range(0,d):
        current_dist = random.betavariate(number_of_times_reward_1[i]+1,
                                          number_of_times_reward_0[i]+1)
        if current_dist > max_dist:
            max_dist = current_dist
            ad = i
    
    ad_display.append(ad)
    
    click_reward = simulation_dataset.values[n,ad]
    
    if click_reward==1:
        number_of_times_reward_1[ad]+=1
    else:
        number_of_times_reward_0[ad]+=0
    
    
plt.hist(ad_display)
plt.show()    