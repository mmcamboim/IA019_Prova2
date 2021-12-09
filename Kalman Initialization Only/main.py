# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 08:47:09 2021

@author: mcamboim
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from kf import KF

plt.close('all')

# Falta só substituir o sistema pelo de AOKI, depois que eu arrumar
with open(r'system.pickle','rb') as file:
    [Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER] = pickle.load(file)
  
xmean_est = np.zeros((500,2))
xvar_inf_est = np.zeros((500,2))
xvar_est = np.zeros((500,2))
y_est = np.zeros((500,2))

kf = KF(Ak,Bk,Ck,yk_t[0].reshape(2,1),W_STD,V_STD)
xvar_inf_est[0] = kf.pk0_inf
xvar_est[0]


for i in range(1,500):
    kf.execute(yk_t[i].reshape(2,1)) 
    
    xmean_est[i,0] = kf.xk0_mean
    xmean_est[i,1] = kf.xk1_mean
    xvar_est[i,0] = kf.pk0
    xvar_est[i,1] = kf.pk1
    xvar_inf_est[i,0] = kf.pk0_inf
    xvar_inf_est[i,1] = kf.pk1_inf
    
    y_est[i,0] = kf.yk0_est
    y_est[i,1] = kf.yk1_est
    
    


plt.figure()
plt.plot(xmean_est[:,0])
plt.plot(xk_t[:,0])
plt.plot(xmean_est[:,1])
plt.plot(xk_t[:,1])

plt.figure()
plt.plot(y_est[:,0])
plt.plot(yk_t[:,0])
plt.plot(y_est[:,1])
plt.plot(yk_t[:,1])

plt.figure()
plt.plot(xvar_est[:,0])
plt.plot(xvar_est[:,1])

plt.figure()
plt.plot(xvar_inf_est[:,0])
plt.plot(xvar_inf_est[:,1])