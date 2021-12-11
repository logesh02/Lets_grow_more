#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE INTERNSHIP AT LETS GROW MORE

# # ADVANCE LEVEL 3 : Develop A Neural Network That Can Read Handwriting:

# ### Import required libraries:

# In[104]:


import tensorflow as tsf
from numpy import unique,argmax
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np


# ### Load the dataset

# In[105]:


mnist = tsf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()


# ### Reshaping the training and testing dataset

# In[106]:


x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))


# ### Normalize the value of pixels in images

# In[107]:


x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


# In[108]:


fig = plt.figure(figsize = (9,3))
for i in range(20):
    ax = fig.add_subplot(2,10,i+1,xticks =[],yticks = [])
    ax.imshow(np.squeeze(x_train[i]),cmap = 'spring')
    ax.set_title(y_train[i])


# ### Determine the shape of input image

# In[109]:


img_shape = x_train.shape[1:]
img_shape


# ### Define the model

# In[110]:


import tensorflow as tfs
model = tsf.keras.models.Sequential([tsf.keras.layers.Flatten(input_shape=(28, 28)),

  tsf.keras.layers.Dense(128, activation='relu'),

  tsf.keras.layers.Dropout(0.2),

  tsf.keras.layers.Dense(10)])


# In[111]:


model.summary()


# In[112]:


get_ipython().system('pip install graphviz ')


# In[113]:


get_ipython().system('pip install pydot')


# In[114]:


get_ipython().system('pip install keras')


# In[ ]:


predictions = model(x_train[:1]).numpy()

predictions


# In[ ]:


tsf.nn.softmax(predictions).numpy()


# ### Compiling The Model¶
# 

# In[ ]:


loss_fn = tsf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',loss=loss_fn, metrics=['accuracy'])


# ### Train the model

# In[ ]:


model.fit(x_train, y_train, epochs=5)


# ### model evaluation

# In[ ]:


model.evaluate(x_test,  y_test, verbose=5)


# ### Probablity of the model¶
# 

# In[ ]:


probability_model = tsf.keras.Sequential([ model, tsf.keras.layers.Softmax() ])
probability_model(x_test[:5])


# ### Testing the model¶
# 

# In[132]:


img = x_train[3]
plt.imshow(np.squeeze(img) ,cmap='gray')
plt.show()


# In[134]:


img= img.reshape(1, img.shape[0],img.shape[1],img.shape[2])
p= model.predict([img])
print("predicted is : {}".format(argmax(p)))

