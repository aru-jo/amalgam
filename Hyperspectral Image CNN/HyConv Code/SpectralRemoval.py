
# coding: utf-8

# ## Import statements and Load Spatially Minimised Array

# In[13]:


import numpy as np 
loaded_cut = np.load('cutarray.npy')
png_cut = np.load('cutGT.npy')
max_rows_index = loaded_cut.shape[0]
max_column_index = loaded_cut.shape[1]
max_spectral_index = loaded_cut.shape[2]


# ## Define Function to Remove Irrelevant Spectral Bands [Absorption]

# The Methane Absorption Spectra lies in the range of 7.5-8 micron, however HyTES images have wavelengths from 7.5-13 micron. The irrelevant spectral bands are now removed. 

# In[9]:


def remove_bands(X,Y):
    mask = list(range(X, Y))
    REM_ARRAY = np.delete(loaded_cut, mask, 2)
    return REM_ARRAY


# In[14]:


loaded_cut = remove_bands(0,64)
max_spectral_index = loaded_cut.shape[2]


# ## Define Function to Remove Irrlevant Spectral Bands [Correlation]

# In[15]:


def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result


# In[30]:


#loaded_cut = loaded_cut.transpose(2,0,1).reshape(1,(max_rows_index+1)*(max_column_index+1))
threshold = 0.70
mask = list(range(187,193))
for band_i in range(0,max_spectral_index):
    band_j = band_i+1
    cf = correlation_coefficient(loaded_cut[band_i],loaded_cut[band_j])
    if(cf>=threshold):
        mask.append(cf)
loaded_cut = remove_bands(int(mask[0]),int(mask[6]))


# In[32]:


loaded_cut.shape

