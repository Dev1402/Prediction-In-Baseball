#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[26]:


plt.subplot(2,2,1)
a = np.arange(1,100,2)
b = np.arange(1,100,2)
plt.plot(a,b)
 
plt.subplot(2,2,4)
a = np.arange(1,100,2)
b = np.arange(1,100,2)
plt.plot(a,b)

plt.subplot(2,2,3)
a = np.array([4,10])

plt.plot(a,b)
plt.show()


# In[27]:


a = np.linspace(1,10,12)
a


# In[30]:


a = np.array([1,10])
a

