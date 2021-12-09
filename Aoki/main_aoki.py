# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:27:35 2021

@author: mcamboim
"""

import numpy as np
import pickle

from scipy import linalg as la
# ARRRUMAR

with open(r'system.pickle','rb') as file:
    [Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER] = pickle.load(file)
    
k = 250
i = 1 + 1
j = 9
# Colocar validações do valor de k, i, k

line_dim = yk_t.shape[1] * i
colu_dim = j 
yk_plus  = np.zeros((line_dim,colu_dim))
yk_minus = np.zeros((line_dim,colu_dim))


# Arrumar isso depois ==============================
for line in range(i):
    for column in range(j):
        yk_plus[2*line, column] = yk_t[k + line + column + 1, 0]
        yk_plus[2*line + 1, column] = yk_t[k + line + column + 1, 1]
        
for line in range(i):
    for column in range(j):
        yk_minus[2*line, column] = yk_t[k - line + column, 0]
        yk_minus[2*line + 1, column] = yk_t[k - line + column, 1]
temp = yk_plus @ yk_minus.T / j

H_lambda = temp[:-2,:-2]
H_lambda_up = temp[2:,:-2]
lambda_yy00 = (yk_minus @ yk_minus.T / j)[:2,:2]

#==============================

U, sdiag, VH = np.linalg.svd(H_lambda)
S = np.zeros((H_lambda.shape[0], H_lambda.shape[1]))
np.fill_diagonal(S, sdiag)
V = VH.T.conj()

## COMPACT
U = U[:,:2]
S = S[:2,:2]
V = V[:2,:]

omicron = U @ np.sqrt(S)
omega = np.sqrt(S) @ V

A = np.linalg.pinv(omicron) @ H_lambda_up @ np.linalg.pinv(omega)
C = H_lambda[:2,:] @ np.linalg.pinv(omega)
M1 = np.linalg.pinv(omicron) @ H_lambda[:,:2]
#DELTA = lambda_yy00 - C @ S @ C.T
#K = (M1 - A @ S @ C.T) @ np.linalg.inv(DELTA)
ric = la.solve_discrete_are(a = A.T, b = C.T, q = np.zeros((2,2)), r = -lambda_yy00, e=None, s=-M1, balanced=True)
DELTA = lambda_yy00 - C @ ric @ C.T
K = (M1 - A @ ric @ C.T) @ np.linalg.inv(DELTA)



with open(r'C:\Users\mcamboim\Documents\UNICAMP\IA019 - Realização e Predição de Séries Temporais Multivariáveis no Espaço de\Provas\Prova 2\Github\Kalman Initialization + Aoki\aoki_out.pickle','wb') as file:
    pickle.dump([A,C,M1,DELTA,K],file)
    
A @ ric @ A.T + (M1 - A @ ric @ C.T ) @ np.linalg.inv( lambda_yy00 - C @ ric @ C.T ) @ (M1 - A @ ric @ C.T).T
