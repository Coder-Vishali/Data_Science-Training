# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

simulation_dataset = pd.read_csv("Ads ClickThroughRate.csv")
N = 10000       # number or rounds
d = 10          # number of Ads

number_of_times_ad_displayed = [0]*d
sum_of_rewards = [0]*d
ad_displayed = []

for n in range(0,______):
    max_ucb=0    
    for i in range(0,d):
        if _______________________:
            average_reward = ____________________________
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_times_ad_displayed[i])
            current_ucb = ______________________
        else:
            current_ucb = _____________
            
        if current_ucb > max_ucb:
            max_ucb = ________
            ad = __________
        
    ad_displayed.append(ad)    
    number_of_times_ad_displayed[ad] = number_of_times_ad_displayed[ad]+1    
    cliked_reward = _________________________
    sum_of_rewards[ad] = sum_of_rewards[ad] + cliked_reward    
        
plt.hist(ad_displayed)
plt.show()
        
        
        
        














