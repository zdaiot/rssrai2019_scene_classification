# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:22:37 2019

@author: zdaiotLab
"""
import matplotlib.pyplot as plt

with open('checkpoint/log.txt') as f:
    data = f.readlines()

train_loss, val_loss = [],[]
data.pop(0)
for x in data:
    x = x.strip().split('\t')
    train_loss.append(round(eval(x[1]),4))
    val_loss.append(round(eval(x[2]),4))

plt.plot(range(len(data)), train_loss, label='train_loss')
plt.plot(range(len(data)), val_loss, label='val_loss')
plt.legend()
plt.show()