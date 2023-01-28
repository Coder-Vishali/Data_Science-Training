# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:28:54 2022
Upper Cofidence bound
@author: TSE
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

simulation_dataset = pd.read_csv("Ads ClickThroughRate.csv")
N=10000         # number of rounds 
d=10            # number of Ads
                                          #  Ad0 Ad1 Ad2 Ad3 Ad4 Ad5 Ad6 
number_of_times_ad_displayed = [0]*d      # [0,  0,  0,  0,  0,  0,0,0,0,0]
sum_of_rewards = [0]*d
ad_displayed = []


for n in range(0,N):          # Loop for Rounds
    max_ucb = 0
    for i in range(0,d):      # Loop for Ads
        if number_of_times_ad_displayed[i]>0:
            average_reward = sum_of_rewards[i]/number_of_times_ad_displayed[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_times_ad_displayed[i])
            current_ucb = average_reward + delta_i           
        else:
            current_ucb = 1e400       # this will be given to Ads who have not got even a single chance            

        if current_ucb > max_ucb:
            max_ucb = current_ucb
            ad = i 
        
    ad_displayed.append(ad)   
    number_of_times_ad_displayed[ad] = number_of_times_ad_displayed[ad] + 1 
    """Ad ad in round n has a click or no-click ??"""
    
    clicked_reward = simulation_dataset.values[n,ad]
    
    sum_of_rewards[ad] = sum_of_rewards[ad] + clicked_reward
    
plt.hist(ad_displayed)    
plt.show()











    













