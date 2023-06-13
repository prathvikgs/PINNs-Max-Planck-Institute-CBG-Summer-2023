#!/usr/bin/env python
# coding: utf-8

# # Simple pde

# **Prathvik G S**

# In[23]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl 
import random

print(tf.__version__)


# # equations

# $$\frac{\partial{u}}{\partial{t}}=1$$
# $$\frac{\partial{u}}{\partial{x}}=-2xe^{-x^2}$$
# $$u(x,0)=e^{-x^2}$$
# $$u(-1,t)=t+e^{-1}$$
# $$u(1,t)=t+e^{-1}$$
# Analytical solution is
# $$u(x,t)=t+e^{-x^2}$$

# In[24]:


def real(x,t):
    return np.e**(-(x)**2)+t


# # The architecture

# In[25]:


tf.keras.backend.set_floatx('float32')

#Define the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(40, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(40, activation='tanh'),
     tf.keras.layers.Dense(40, activation='tanh'),
    tf.keras.layers.Dense(40, activation='tanh'),
    tf.keras.layers.Dense(1)
])


# # Custom loss function 

# In[38]:


#loss function for the PINN
def loss(x_pde,x_ini,x_b):
    
    with tf.GradientTape(persistent=True) as tape:
        
        #loss from the PDE
        x,t=tf.unstack(x_pde,axis=1)
        tape.watch(t)
        tape.watch(x)
        
        u_pred= model(tf.stack((x,t),axis=1))
        
        dudt  =  tape.gradient(u_pred,t)
        dudx  =  tape.gradient(u_pred,x)
        loss1 =  tf.reduce_mean(tf.square(dudt-1))
        loss2 =  tf.reduce_mean(tf.square(dudx+(2*x*np.e**(-x**2))))
        
        #boundary loss
        xb,tb=tf.unstack(x_b,axis=1)
        u_predb = model(tf.stack((xb,tb),axis=1))
        u_b= real(xb,tb)
        lossb=tf.reduce_mean(tf.square(u_predb-u_b))
        
        
        #initial loss
        xi,ti=tf.unstack(x_ini,axis=1)
        u_predi = model(tf.stack((xi,ti),axis=1))
        u_ini= real(xi,ti)
        lossi=tf.reduce_mean(tf.square(u_predi-u_ini))
        
    return  1000*loss1 + 10*lossi


# In[39]:


# Define the grid size
N = 30
x_0=-1
x_L=1
t_0=0
t_L=1

# Define the coordinates of the grid points
x = np.linspace(x_0, x_L, N)
t = np.linspace(t_0, t_L, N)

nx, ny = np.meshgrid(x, t, indexing='ij')
nx1=np.reshape(nx,-1)
ny1=np.reshape(ny,-1)
#nx=2*np.random.rand(100000,1)-1
#ny=2*np.random.rand(100000,1)-1
nxny = np.concatenate([nx1[:, None], ny1[:, None]], axis=1)
x_pde=tf.convert_to_tensor(nxny, dtype=tf.float32, dtype_hint=None, name=None)


# In[40]:


'initial'
x = np.linspace(x_0, x_L, 200)
y = np.zeros((200,1))
nx1=np.reshape(x,-1)
ny1=np.reshape(y,-1)
x_ini = np.concatenate([nx1[:, None], ny1[:, None]], axis=1)
x_ini=tf.convert_to_tensor(x_ini, dtype=tf.float32, dtype_hint=None, name=None)

'boundary'

x = -1*np.ones((200,1))
y = np.linspace(t_0, t_L, 200)
nx1=np.reshape(x,-1)
ny1=np.reshape(y,-1)
x_b = np.concatenate([nx1[:, None], ny1[:, None]], axis=1)
x_b=tf.convert_to_tensor(x_b, dtype=tf.float32, dtype_hint=None, name=None)


# In[ ]:


# Define the optimizer

optimizer = tf.keras.optimizers.Adam()
epochs=1000
# Train the model
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_value = loss(x_pde,x_ini,x_b)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 1 == 0:
        #print(f"Epoch {epoch}, loss={loss_value:.6f}")
        print("epoch {}/{}, loss={:.10f}".format(epoch+1,epochs,  loss_value), end="\r")


# In[42]:


# Define the coordinates of the grid points
x = np.linspace(x_0, x_L, N)
y = np.linspace(t_0, t_L, N)

nx, ny = np.meshgrid(x, y, indexing='ij')
nx, ny = np.meshgrid(x, y, indexing='ij')
nx1=np.reshape(nx,-1)
ny1=np.reshape(ny,-1)
nxny = np.concatenate([nx1[:, None], ny1[:, None]], axis=1)

X=model(nxny)
p=tf.unstack(X,axis=1)


# # Solution

# In[43]:


p=tf.reshape(p,(N,N))
fig, ax = plt.subplots(1, 1)
c=plt.pcolormesh(nx,ny,p)
plt.colorbar()
plt.title('PINN Solution')
plt.show()

p_t=real(nxny[:,0],nxny[:,1])
p_t=np.reshape(p_t,(N,N))
fig, ax = plt.subplots(1, 1)
c=plt.pcolormesh(nx,ny, p_t)
plt.title("Analytical solution")
plt.colorbar()
plt.show()

fig, ax = plt.subplots(1, 1)
c=plt.pcolormesh(nx,ny, (((p_t-p)/p_t)**2)*100)
plt.colorbar()
plt.title("per centage Error in %")
plt.show()

