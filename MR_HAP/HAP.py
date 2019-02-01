# -*- coding: utf-8 -*-
"""
    Implementation of the Hierarchical Affinity Propagation
    
"""

import numpy as np
from sklearn.metrics import euclidean_distances

class HAP():
    
    def __init__(self,data, L, N):
        self.L = L
        self.N =  N
        self.data = data
        self.affinity_matrix = -euclidean_distances(data, squared=True)
        preference =  np.random.rand(N)*(-10**6)
        np.fill_diagonal(self.affinity_matrix, preference)
        
        self.A = np.zeros((L,N,N), dtype = 'float32') 
        self.S = np.zeros((L,N,N), dtype = 'float32') 
        self.S[0,: ,:] = self.affinity_matrix

        self.R = np.zeros((L,N,N), dtype = 'float32') 
        #R[0,: ,:] = np.array([[1,2,3],[4,5,6], [7,8,9]])

        self.damping = 0.5
        self.kappa = 0.1

        self.TAU = np.multiply(np.ones((L,N), dtype = 'float32'), np.inf)
        #print(tau)
        self.PHI = np.zeros((L,N), dtype = 'float32')
        self.C = np.zeros((L,N), dtype = 'float32')
        self.E = np.zeros((L,N), dtype = 'float32')

        self.tmp = np.zeros((N,N), dtype = 'float32')
        self.tmp2 = np.zeros((N,N), dtype = 'float32')

        self.Y = np.zeros(N, dtype = 'float32')

    def update_r(self, l) : 

        N = self.N
        ind = np.arange(N)

        self.tmp = np.zeros((N,N))
        #Add alpha_ik and s_ik and put it in tmp
        np.add(self.A[l, :, : ], self.S[l, :, :], self.tmp)

        # Get the maximim value on row and save the vector apart
        I = np.argmax(self.tmp, axis=1)
        self.Y = self.tmp[ind, I].copy()
        self.Y *= (-1)

        # Set -inf to maximum values and get the second maximum values.
        self.tmp[ind, I] = -np.inf
        Y2 = np.max(self.tmp, axis=1)
        Y2 *= (-1)

        tmp2 = np.min((self.TAU[l, :, None],self.Y[:, None]), axis=0)
        np.add(self.S[l,: ,:], tmp2, self.tmp)

        self.tmp[ind, I] = self.S[l, ind, I] + np.min((self.TAU[l,:],Y2), axis=0)

        # Dampen
        Rl = self.R[l, :,  :].copy()
        self.tmp *= (1 - self.damping)
        Rl *= self.damping
        Rl += self.tmp

        return Rl

    def update_a(self, l):

        N = self.N
        self.tmp = np.zeros((N,N))
        np.maximum(self.R[l, :, :], 0, self.tmp)   
        tmp0 = np.sum(self.tmp, axis=0)
        # On sommes les sur les colonnes meme les indices  i et j (qu'il faudra retirer)

        # On ajoutera les jj ssi ils sont negatifs (ie ils auront compté pour 0 sur la somme ci-dessus)
        self.Y = np.zeros(N)
        np.minimum(np.diag(self.R[l, :, :]), 0, self.Y) 

        # Ils faut aussi retirer tout les indices (i,j)
        Y2 = np.zeros((N,N))
        np.maximum(self.R[l, :, :], 0, Y2)   
        np.fill_diagonal(Y2,0)
        Y2 *= (-1)

        #Y2 at the very end !! It does not depend on i except for Y2 -> tmp - Y2 at the end only !!
        np.add(tmp0, self.Y, tmp0)
        np.add(tmp0, self.PHI[l, :], tmp0)
        np.add(tmp0, self.C[l, :], tmp0)

        # Now, we can substract Y2 (add -Y2)
        np.add(Y2,tmp0[None,:], self.tmp)
        np.minimum(self.tmp,0, self.tmp)

        # Now the diagnonal
        self.tmp2 = self.R[l, :, :].copy()
        np.fill_diagonal(self.tmp2, -np.inf)
        np.maximum(self.tmp2, 0, self.tmp2) 
        self.tmp2 = np.sum(self.tmp2, axis=0)
        np.add(self.tmp2, self.PHI[l, :], self.tmp2)
        np.add(self.tmp2, self.C[l, :], self.tmp2)
        np.fill_diagonal(self.tmp,self.tmp2)

        Al = self.A[l, :, :].copy()
        self.tmp *= 1 - self.damping
        Al *= self.damping
        Al += self.tmp

        return Al

    def update_tau(self, l):   
        

        self.tmp = self.R[l, :, :].copy()

        #makes sure the diagonal will be 0, so that we can sum on all k.
        np.fill_diagonal(self.tmp, 0)
        np.maximum(self.tmp, 0, self.tmp)   

        #Takes sum for each columns (ie for sum function, axis=0)
        tmp2 = np.sum(self.tmp, axis=0)

        np.add(tmp2, np.diag(self.R[l, :, :]), tmp2)   
        np.add(tmp2,             self.C[l, :], tmp2)

        return tmp2

    def update_phi(self, l):

        self.tmp = np.add(self.A[l, :, :] ,self.S[l, :, :])
        return np.max(self.tmp, axis=1)

    def update_c(self, l):

        self.tmp = np.add(self.A[l, :, :], self.R[l, :, :])
        return np.max(self.tmp, axis=1)

    def update_S(self, l, update=False):

        if update == False :
             return self.S[l, :, :].copy()

        self.tmp =  (self.A[l, :, :] + self.R[l, :, :])
        np.fill_diagonal(self.tmp, -np.inf)
        self.Y = np.max(self.tmp, axis=1)
        self.Y*= self.kappa

        np.add(self.S[l, :, :], self.Y[:,None], self.tmp)
        return self.tmp
    
    def fit(self, nb_iteration):
        
        for iteration in range(nb_iteration):
            
            #tmp = np.zeros((N,N))
            for l in range(self.L):

                if l > 0:
                    Rl = self.update_r(l)
                    #self.R[l, :, :] = self.update_r(l)
                    
                if l < (self.L-1):
                    Al = self.update_a(l)
                    #self.A[l, :, :] = self.update_a(l)
                    
                if l < (self.L-1) :

                    t = self.update_tau(l)
                    #self.TAU[(l+1), :] = self.update_tau(l)
                    
                if l > 0 :
                    p = self.update_phi(l)
                    #self.PHI[(l-1), :] = self.update_phi(l)
                
                cl = self.update_c(l)
                #self.C[l,:] = self.update_c(l)
                
                if l < (self.L-1) :
                    self.S[l+1, :, : ] = self.update_S(l, update = False).copy()

                if l > 0:
                    self.R[l, :, :] = Rl.copy()
                    del Rl
                    self.PHI[(l-1), :] = p.copy()
                    del p
                if l < (self.L-1):
                    self.A[l, :, :] = Al.copy()
                    del Al
                    self.TAU[(l+1), :] = t.copy()
                    del t 
                self.C[l,:] = cl.copy()
                del cl

            #print('it :', iteration)
            for l in range(self.L):
                self.temp = np.add(self.A[l, :, :],self.R[l, :, :])
                self.E[l,:] = np.argmax(self.temp, axis=1)
                #print(len(Counter(self.E[l,:] ).keys()))
                #if (len(Counter(self.E[l,:] ).keys())) < 10 :
                #    print(Counter(self.E[l,:] ))

        #print('\n************* e ****************')
        for l in range(self.L):
            self.temp = np.add(self.A[l, :, :],self.R[l, :, :])
            self.E[l,:]  = np.argmax(self.temp, axis=1)
            #print(Counter(self.E[l,:] ))

    