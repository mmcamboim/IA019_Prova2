# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 08:47:35 2021

@author: mcamboim
"""

import numpy as np

class KF:
    def __init__(self,Ak,Bk,Ck,yk0,DELTA):
        self._xk = np.zeros(Ak.shape[1]).reshape(2,1)
        self._Ak = Ak
        self._Bk = Bk
        self._Ck = Ck
        self._Qk = DELTA
        self._Rk = DELTA
        
        # Inicialicao_exata
        self._Pk_inf = np.eye(Ak.shape[1])
        self._Pk_ast = np.zeros((Ak.shape[0],Ak.shape[1]))
        
        self._Fk_inf = self._Ck @ self._Pk_inf @ self._Ck.T
        self._Fk_ast = self._Ck @ self._Pk_ast @ self._Ck.T + self._Rk
        
        self._Mk_inf = self._Pk_inf @ self._Ck.T
        self._Mk_ast = self._Pk_ast @ self._Ck.T
        
        self._Kk0 = self._Ak @ self._Mk_inf @ np.linalg.inv(self._Fk_inf)
        self._Kk1 = self._Ak @ (self._Mk_ast @ np.linalg.inv(self._Fk_inf) - self._Mk_inf @ np.linalg.inv(self._Fk_inf) @ self._Fk_ast @ self._Fk_inf)
        
        self._Lk0 = self._Ak - self._Kk0 @ self._Ck
        self._Lk1 = - self._Kk1 @ self._Ck
        
        self._xk0 = self._Ak @ self._xk + self._Kk0 @ (yk0 - self._Ck @ self._xk)
        self._xk1 = self._Kk1 @ (yk0 - self._Ck @ self._xk)
        
        self.convergency = False
    
    def execute(self,yk):
        self._ek0 = yk - self._Ck @ self._xk0
        if(not self.convergency):
            self._ek1 = - self._Ck @ self._xk1
        else:
            self._ek1 = np.zeros((2,1))
        self._ek = self._ek0 + self._ek1
        
        if(not self.convergency):
            Pk_inf_temp = self._Ak @ self._Pk_inf @ self._Lk0.T
        else:
            Pk_inf_temp = np.zeros((2,2))
        Pk_ast_temp = self._Ak @ self._Pk_inf @ self._Lk1.T + self._Ak @ self._Pk_ast @ self._Lk0.T + self._Bk @ self._Qk @ self._Bk.T
        self._Pk_inf, self._Pk_ast = Pk_inf_temp, Pk_ast_temp
        self.convergency = np.allclose(self._Pk_inf,np.zeros((2,2)))
        
        
        self._Fk_inf = self._Ck @ self._Pk_inf @ self._Ck.T
        self._Fk_ast = self._Ck @ self._Pk_ast @ self._Ck.T + self._Rk
        
        self._Mk_inf = self._Pk_inf @ self._Ck.T
        self._Mk_ast = self._Pk_ast @ self._Ck.T
        
        if(not self.convergency):
            self._Kk0 = self._Ak @ self._Mk_inf @ np.linalg.inv(self._Fk_inf)
        else:
            self._Kk0 = self._Ak @ self._Mk_ast @ np.linalg.inv(self._Fk_ast)
        if(not self.convergency):
            self._Kk1 = self._Ak @ (self._Mk_ast @ np.linalg.inv(self._Fk_inf) - self._Mk_inf @ np.linalg.inv(self._Fk_inf) @ self._Fk_ast @ self._Fk_inf)
        else: 
            self._Kk1 = np.zeros((2,2))
        
        self._Lk0 = self._Ak - self._Kk0 @ self._Ck
        self._Lk1 = - self._Kk1 @ self._Ck


        self._xk0 = self._Ak @ self._xk0 + self._Kk0 @ self._ek0
       
        if(not self.convergency):
            self._xk1 = self._Ak @ self._xk1 + self._Kk0 @ self._ek1 + self._Kk1 @ self._ek0
        else:
            self._xk1 = np.zeros((2,1))
            
        self._xk = self._xk0 + self._xk1
    
    @property
    def xk0_mean(self) -> float:
        return self._xk[0,0]
    
    @property
    def xk1_mean(self) -> float:
        return self._xk[1,0]
    
    @property
    def yk0_est(self) -> float:
        return (self._Ck @ self._xk)[0,0]
    
    @property
    def yk1_est(self) -> float:
        return (self._Ck @ self._xk)[1,0]
    
    @property
    def pk0_inf(self) -> float:
        return self._Pk_inf[0,0]
    
    @property
    def pk1_inf(self) -> float:
        return self._Pk_inf[1,1]
    
    @property
    def pk0(self) -> float:
        return self._Pk_ast[0,0]
    
    @property
    def pk1(self) -> float:
        return self._Pk_ast[1,1]
    