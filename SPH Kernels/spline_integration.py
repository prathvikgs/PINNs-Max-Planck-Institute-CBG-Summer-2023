#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import time
from sympy import symbols, integrate, Abs
import numpy as np
from scipy.integrate import quad
import sympy as sp


# In[3]:


def kernel(z,eps):
    alphad = 2.0/eps
    z = np.abs(z)
    if z < 0.5*eps:
        return alphad*(2.0/3.0 - (2.0*z/eps)**2 + 0.5*(2.0*z/eps)**3)
    elif z >= 0.5*eps and z < eps:
        return alphad*(((1.0/6.0)*(2.0-2.0*z/eps)**3))
    else:
        return 0.0


# # 1.5eps <d<=2eps
# d=(xj-xi)

# In[4]:

def subtract_kernel_product(xi, xj ,l,r,eps):
    def k(x):
        return kernel(abs(x - xi),eps) * kernel(abs(x - xj),eps)
    result, error = quad(k, l, r)
    return result


# In[5]:


x, d, a, e = symbols('x d a e')
f = (2/e)*(((1.0/6.0)*(2.0-2.0*x/e)**3))*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
part2 = integrate(f, (x,d-e ,e))

part2_np = sp.lambdify((d,e),part2)


# # eps <=d<1.5 eps
# d=xj-xi

# In[6]:


x, d, a, e = symbols('x d a e')
f1=(2/e)*(2.0/3.0 - (2.0*(x)/e)**2 + 0.5*(2.0*(x)/e)**3)*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
q1pq2n=integrate(f1, (x,d-e ,0.5*e))

f2=(2/e)*(((1.0/6.0)*(2.0-2.0*x/e)**3))*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
q2pq2n=integrate(f, (x,0.5*e ,d-0.5*e))

f3=(2/e)*(((1.0/6.0)*(2.0-2.0*(x)/e)**3))*(2/e)*(2.0/3.0 - (2.0*((d-x))/e)**2 + 0.5*(2.0*(d-x)/e)**3)
q2pq1n=integrate(f3, (x,d-0.5*e ,e))

part3=q1pq2n+q2pq2n+q2pq1n

part3_np = sp.lambdify((d,e),part3)


# # 0.5eps<d<eps

# In[7]:


x, d, a, e = symbols('x d a e')
f11=(2/e)*(2.0/3.0 - (2.0*(-x)/e)**2 + 0.5*(2.0*(-x)/e)**3)*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
q1nq2n=integrate(f11, (x,d-e ,0))

f12=(2/e)*(2.0/3.0 - (2.0*(x)/e)**2 + 0.5*(2.0*(x)/e)**3)*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
q1pq2n=integrate(f12, (x,0 ,d-0.5*e))

f2=(2/e)*(2.0/3.0 - (2.0*(x)/e)**2 + 0.5*(2.0*(x)/e)**3)*(2/e)*(2.0/3.0 - (2.0*(d-x)/e)**2 + 0.5*(2.0*(d-x)/e)**3)
q1pq1n=integrate(f2, (x,d-0.5*e,0.5*e))

f3=(2/e)*(((1.0/6.0)*(2.0-2.0*(x)/e)**3))*(2/e)*(2.0/3.0 - (2.0*((d-x))/e)**2 + 0.5*(2.0*(d-x)/e)**3)
q2pq1n=integrate(f3, (x,0.5*e ,e))

part4=q1nq2n+q1pq2n+q1pq1n+q2pq1n

part4_np = sp.lambdify((d,e),part4)


# #  0<=d<0.5eps

# In[8]:


x, d, a, e = symbols('x d a e')
f1=(2/e)*(((1.0/6.0)*(2.0-2.0*(-x)/e)**3))*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
q2nq2n=integrate(f1, (x,d-e ,-0.5*e))

f2=(2/e)*(2.0/3.0 - (2.0*(-x)/e)**2 + 0.5*(2.0*(-x)/e)**3)*(2/e)*(((1.0/6.0)*(2.0-2.0*(d-x)/e)**3))
q1nq2n=integrate(f2, (x,-0.5*e ,d-0.5*e))

f3=(2/e)*(2.0/3.0 - (2.0*(-x)/e)**2 + 0.5*(2.0*(-x)/e)**3)*(2/e)*(2.0/3.0 - (2.0*(d-x)/e)**2 + 0.5*(2.0*(d-x)/e)**3)
q1nq1n=integrate(f3, (x,d-0.5*e,0))

f4=(2/e)*(2.0/3.0 - (2.0*(x)/e)**2 + 0.5*(2.0*(x)/e)**3)*(2/e)*(2.0/3.0 - (2.0*(d-x)/e)**2 + 0.5*(2.0*(d-x)/e)**3)
q1pq1n=integrate(f4, (x,0,d))

f5=(2/e)*(2.0/3.0 - (2.0*(x)/e)**2 + 0.5*(2.0*(x)/e)**3)*(2/e)*(2.0/3.0 - (2.0*(x-d)/e)**2 + 0.5*(2.0*(x-d)/e)**3)
q1pq1p=integrate(f5, (x,d,0.5*e))

f6=(2/e)*(((1.0/6.0)*(2.0-2.0*(x)/e)**3))*(2/e)*(2.0/3.0 - (2.0*((x-d))/e)**2 + 0.5*(2.0*(x-d)/e)**3)
q2pq1p=integrate(f6, (x,0.5*e ,d+0.5*e))

f7=(2/e)*(((1.0/6.0)*(2.0-2.0*x/e)**3))*(2/e)*(((1.0/6.0)*(2.0-2.0*(x-d)/e)**3))
q2pq2n=integrate(f7, (x,d+0.5*e ,e))

part5=q2nq2n+q1nq2n+q1nq1n+q1pq1n+q1pq1p+q2pq1p+q2pq2n
part5_np = sp.lambdify((d,e),part5)


# In[9]:


eps=0.001
def kernel(z,eps):
    alphad = 2.0/eps
    z = np.abs(z)
    if z < 0.5*eps:
        return alphad*(2.0/3.0 - (2.0*z/eps)**2 + 0.5*(2.0*z/eps)**3)
    elif z >= 0.5*eps and z < eps:
        return alphad*(((1.0/6.0)*(2.0-2.0*z/eps)**3))
    else:
        return 0.0
    
import numpy as np
from scipy.integrate import quad

def integrate_kernel_product(xi, xj ,l,r):
    def k(x):
        return kernel(abs(x - xi),eps) * kernel(abs(x - xj),eps)
    result, error = quad(k, l, r)
    return result


# In[11]:


def ikp(xi,xj,eps):
    if(xi>xj):
        temp=xi
        xi=xj
        xj=temp
        
    eps_val = eps
    alpha_val = 2/eps_val
    d_val=abs(xj-xi)
    subtract_are=0.
    if(xj-eps<-1):
        subtract_are=subtract_kernel_product(xi, xj ,xj-eps,-1,eps)
    elif(xi+eps>1):
        subtract_are=subtract_kernel_product(xi, xj ,1,xi+eps,eps)
        
    if(d_val>=2*eps):
        return 0+subtract_are
    elif(d_val<2*eps and d_val>=1.5*eps):
        return part2_np(d_val,eps)-subtract_are
    elif(d_val<1.5*eps and d_val>=eps):
        return part3_np(d_val,eps)-subtract_are
    elif(d_val<eps and d_val>=0.5*eps):
        return part4_np(d_val,eps)-subtract_are
    elif(d_val<0.5*eps and d_val>=0):
        return part5_np(d_val,eps)-subtract_are


# In[12]:
def ikp(xi,xj,eps):
    if(xi>xj):
        temp=xi
        xi=xj
        xj=temp
        
    eps_val = eps
    alpha_val = 2/eps_val
    d_val=abs(xj-xi)
    subtract_are=0.
        
    if(d_val>=2*eps):
        return 0+subtract_are
    elif(d_val<2*eps and d_val>=1.5*eps):
        return part2_np(d_val,eps)-subtract_are
    elif(d_val<1.5*eps and d_val>=eps):
        return part3_np(d_val,eps)-subtract_are
    elif(d_val<eps and d_val>=0.5*eps):
        return part4_np(d_val,eps)-subtract_are
    elif(d_val<0.5*eps and d_val>=0):
        return part5_np(d_val,eps)-subtract_are
