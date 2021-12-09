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

#BEST
#i = 4 + 1
#j = 6 

#i = 1 + 1
#j = 9 
   
with open(r'system.pickle','rb') as file:
    [_,_,_,xk_t,yk_t,_,_,ITER] = pickle.load(file)

with open(r'aoki_out.pickle','rb') as file:
    [A,C,M1,DELTA,K] = pickle.load(file)
  
xmean_est = np.zeros((ITER,2))
xvar_inf_est = np.zeros((ITER,2))
xvar_est = np.zeros((ITER,2))
y_est = np.zeros((ITER,2))

kf = KF(Ak=A,Bk=K,Ck=C,yk0=yk_t[0].reshape(2,1),DELTA=DELTA)
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


# Sistema
plt.rcParams['axes.linewidth'] = 2.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(xk_t[:,0],c='b',lw=2)
plt.plot(xk_t[:,1],c='r',lw=2)
plt.legend(['$xk_1$','$xk_2$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('xk []')

plt.subplot(2,1,2)
plt.plot(yk_t[:,0],c='b',lw=2)
plt.plot(yk_t[:,1],c='r',lw=2)
plt.legend(['$yk_1$','$yk_2$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('yk []')

plt.tight_layout()

# Estados
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(xk_t[:,0],c='b',lw=2)
plt.plot(-xmean_est[:,0],c='r',lw=2)
plt.legend(['$xk_{1real}$','$xk_{1est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('xk []')

plt.subplot(2,1,2)
plt.plot(xk_t[:,1],c='b',lw=2)
plt.plot(xmean_est[:,1],c='r',lw=2)
plt.legend(['$xk_{2real}$','$xk_{2est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.ylabel('xk []')
plt.xlabel('(b)\n Iteração [N]\n')

plt.tight_layout()


# Saída
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(yk_t[:,0],c='b',lw=2)
plt.plot(y_est[:,0],c='r',lw=2)
plt.legend(['$yk_{1real}$','$yk_{1est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('yk []')

plt.subplot(2,1,2)
plt.plot(yk_t[:,1],c='b',lw=2)
plt.plot(y_est[:,1],c='r',lw=2)
plt.legend(['$yk_{2real}$','$yk_{2est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.ylabel('yk []')
plt.xlabel('(b)\n Iteração [N]\n')

plt.tight_layout()


# Covariâncias Infinita
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(xvar_inf_est[:,0],c='b',lw=2)
plt.xlim([-5,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$P_{inf}^{0,0}$(k|k-1) []')

plt.subplot(2,1,2)
plt.plot(xvar_inf_est[:,1],c='b',lw=2)
plt.xlim([-5,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$P_{inf}^{1,1}$(k|k-1) []')
plt.tight_layout()


# Covariâncias Finitas
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(xvar_est[:,0],c='b',lw=2)
plt.xlim([-5,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$P_{*}^{0,0}$(k|k-1) []')

plt.subplot(2,1,2)
plt.plot(xvar_est[:,1],c='b',lw=2)
plt.xlim([-5,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$P_{*}^{1,1}$(k|k-1) []')
plt.tight_layout()
"""
"""