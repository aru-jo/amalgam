
# coding: utf-8

# ## Import Statements

# In[1]:


import pysptools.util as util
import pandas as pd
import matplotlib.pyplot as plt
from spectral import *
import numpy as np
import pysptools.eea as eea
import spectral.io.envi as envi
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import pysptools.classification as cls
import pysptools.noise as ns
import pysptools.skl as skl
import timeit


# In[2]:


im = plt.imread('methane.png')


# ![title](methane.png)

# ## Import data

# In[4]:


#data_file = r'C:\Users\Dr.RSK\Desktop\Hyperspectral-CNN\AlisoCanyon1\aliso1.hdr'
#data, header = util.load_ENVI_file(data_file)


# ## Dimesnsions before Spatial Removal

# In[111]:


#data.shape


# ## Setting 2D ground truth map

# In[3]:


GT_Dimensions =(5120,512) 


# ## Set zeroes for Pixels which are not methane in Ground Truth

# In[4]:


GT_Array = np.zeros(GT_Dimensions)


# In[5]:


GT_Array


# ## Determine Merthane Presence ->Flip to 1

# In[6]:



#From GT: 

Y = [4210,4186,4185,4155,4119,4118,4118,4113,4109,4123,4094,4022,4021,4072,4072,4008,4053,4011,4008,3993,3988,3990,3982,3978,3973,3972,3987,3977,3943,3943,3930,3940,3956,3942,3932,3959,3953,3941,3929,3944,3963,3984,3965,3987,3985,3987 ,3987,3938,3980,3972,3989,3940,3942,3975]
X = [27,46,49,31,36,36,40,54,76,149,95,257,258,93,102,187,80,185,185,210,183,210,218,219,233,234,184,202,236,235,236,199,207,234,236,204,205,234,238,113,125,141,123,140,141,155,182,200,205,232,183,200,218,187]

for i in range(0,54):
    x_Dim = X[i]
    y_Dim = Y[i]
    GT_Array[y_Dim][x_Dim]=1

print(GT_Array[4210][27])
print(GT_Array[3975][187])


# ## Reflectance values for all pixels across all spectral bands

# In[123]:


print(data)


# ## Header - Meta Data

# In[124]:


print(header)


# ## NFINDR algorithm to detect endmember spectra of entire image 

# In[125]:


def get_endmembers_nfindr(data, header):
    print('Endmembers extraction with NFINDR')
    nfindr = eea.NFINDR()
    U = nfindr.extract(data, 2, maxit=5, normalize=True, ATGP_init=True)
    nfindr.display(header, suffix='Cuprite Endmembers')
    return U


# ## Define Spatial Cut Functions

# In[44]:


del_arr = []
del_arr_Y =[]


# In[45]:


def array_cut_Y(X,Y):
    for i in range(X,Y):
        del_arr_Y.append(i)
    return del_arr_Y


# In[46]:


def array_cut(X,Y):
    for i in range(X,Y):
        del_arr.append(i)
    return del_arr


# ## Assign data to holder variable

# In[45]:


before_cut = data


# ## Cut along X dimension 

# In[47]:


del_arr = array_cut(0,3912)


# In[48]:


del_arr = array_cut(4221,5120)


# In[85]:


after_cut = np.delete(before_cut,del_arr,0)


# In[49]:


im=np.delete(im,del_arr,0)


# In[50]:


im.shape


# In[86]:


after_cut.shape


# ## Cut along Y dimension 

# In[51]:


del_arr_1 = array_cut_Y(270,512)


# In[88]:


after_cut_1 = np.delete(after_cut,del_arr_1,1)


# In[52]:


im=np.delete(im,del_arr_Y,1)


# In[89]:


after_cut_1.shape


# In[53]:


im.shape


# In[54]:


imshow(im)


# ## Save array after Spatial Removal 

# In[8]:


np.save('cutarray.npy',after_cut_1)


# In[55]:


np.save('cutGT.npy',im)


# In[61]: