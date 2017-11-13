# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:16:14 2017

@author: Teo
"""

import pandas as pd
df_complete = pd.read_csv('data/amazon_step2_unseen.csv')
df_complete.to_pickle('amazon_step2_unseen.pkl')