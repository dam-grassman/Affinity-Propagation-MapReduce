import numpy as np
from math import ceil
from sklearn.metrics import euclidean_distances
from collections import Counter
import multiprocessing


class MRHAP():  
    
    #Same initialization as in the standard HAP
    def __init__(self, data, L, N, n_jobs):
        self.L = L
        self.N =  N
        self.data = data
        self.n_jobs = n_jobs
        self.affinity_matrix = -euclidean_distances(data, squared=True)
        preference =  np.random.rand(N)*(-10**6)
        np.fill_diagonal(self.affinity_matrix, preference)
        
        self.A = np.zeros((L,N,N), dtype = 'float32') 
        self.S = np.zeros((L,N,N), dtype = 'float32') 
        self.S[0,: ,:] = self.affinity_matrix
        for l in range(self.L) :
            self.S[l,: ,:] = self.affinity_matrix
        self.R = np.zeros((L,N,N), dtype = 'float32')

        self.damping = 0.5
        self.kappa = 0.1

        self.TAU = np.multiply(np.ones((L,N), dtype = 'float32'), np.inf)
        self.PHI = np.zeros((L,N), dtype = 'float32')
        self.C = np.zeros((L,N), dtype = 'float32')
        self.E = np.zeros((L,N), dtype = 'float32')

        self.pool = multiprocessing.Pool(n_jobs)
        
        
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

######################################## UPDATE FUNCTIONS ##############################################

    def update_tau(self, j,k,l):
        tmp = self.R[l, :, j:k].copy()
        #makes sure the diagonal will be 0, so that we can sum on all k.
        np.fill_diagonal(tmp, 0)
        np.maximum(tmp, 0,tmp)   

        #Takes sum for each columns (ie for sum function, axis=0)
        tmp2 = np.sum(tmp, axis=0)
        np.add(tmp2, np.diag(self.R[l, :,j:k]), tmp2)   
        np.add(tmp2, self.C[l, j:k], tmp2)
        return tmp2    
  

    def update_c(self,i,k,l):
        tmp = np.add(self.A[l, i:k, :], self.R[l, i:k, :])
        return np.max(tmp, axis=1)

    
    def update_r(self,j,k,l): 
        #nodes   
        ind = np.arange(self.N)
        #sous-matrice
        tmp = np.zeros((self.N,k-j))
        #Add alpha_ik and s_ik and put it in tmp
        np.add(self.A[l, :, j:k], self.S[l, :, j:k],tmp)

        # Get the maximim value on row and save the vector apart
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I].copy()
        Y *= (-1)

        # Set -inf to maximum values and get the second maximum values.
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)
        Y2 *= (-1)
        tmp2 = np.min((self.TAU[l, :, None],Y[:, None]), axis=0)
        np.add(self.S[l,: ,j:k], tmp2, tmp)
        tmp[ind, I] = self.S[l, ind, I] + np.min((self.TAU[l,:],Y2), axis=0)

        # Dampen
        Rl = self.R[l, :,  j:k].copy()
        tmp *= (1 - self.damping)
        Rl *= self.damping
        Rl += tmp
        
        return Rl
    
    def update_a(self,i,k, l):
        N = self.N
        #tmp = np.zeros((k-i+1,N))
        tmp = np.zeros((k-i,N))

        np.maximum(self.R[l, i:k, :], 0, tmp)   
        tmp0 = np.sum(tmp, axis=0)
        # On sommes les sur les colonnes meme les indices  i et j (qu'il faudra retirer)

        # On ajoutera les jj ssi ils sont negatifs (ie ils auront compté pour 0 sur la somme ci-dessus)
        Y = np.zeros(k-i)
        np.minimum(np.diag(self.R[l, i:k, :]), 0, Y) 

        # Ils faut aussi retirer tout les indices (i,j)
        Y2 = np.zeros((k-i,N))
        np.maximum(self.R[l, i:k, :], 0, Y2)   
        np.fill_diagonal(Y2,0)
        Y2 *= (-1)

        #Y2 at the very end !! It does not depend on i except for Y2 -> tmp - Y2 at the end only !!
        
        tmp0[i:k] =  np.add(tmp0[i:k],Y)
        
        #np.add(tmp0, Y, tmp0)
        np.add(tmp0, self.PHI[l, :], tmp0)
        np.add(tmp0, self.C[l, :], tmp0)

        # Now, ca cann substract Y2 (add -Y2)
        np.add(Y2,tmp0[None,:], tmp)
        np.minimum(tmp,0, tmp)

        # Now the diagnonal
        tmp2 = self.R[l, i:k, :].copy()
        np.fill_diagonal(tmp2, -np.inf)
        np.maximum(tmp2, 0, tmp2) 
        tmp2 = np.sum(tmp2, axis=0)
        np.add(tmp2, self.PHI[l, :], tmp2)
        np.add(tmp2, self.C[l, :], tmp2)
        np.fill_diagonal(tmp,tmp2)

        Al = self.A[l, i:k, :].copy()
        tmp *= 1 - self.damping
        Al *= self.damping
        Al += tmp
        #print(Al)
        return Al
    
    def update_phi(self,i,k, l):
        tmp = np.add(self.A[l, i:k, :],self.S[l, i:k, :])
        return np.max(tmp, axis=1)
    

#################################### MAPPING Functions ###########################################################

    # MAPPER  which update tau, c and rho - Example-based format as input
    def mapper_A(self,args):
        
        j,k,l, tensor = args
        if tensor == 'tau' and l<self.L-1:
            return ([j,k,l,tensor],self.update_tau(j,k,l))
            
        #elif tensor == "c":
        #    return ([j,k,l,tensor],self.update_c(j,k,l))
        
        elif tensor == "rho" and l > 0:
            return ([j,k,l,tensor],self.update_r(j,k,l))
        else :
            return ([],None)


    # MAPPER B which update alpha and phi - Node-based format as input
    def mapper_B(self, args):

        i,k,l, tensor = args
        if tensor == 'alpha' and l<self.L -1:
            return ([i,k,l,tensor],self.update_a(i,k,l))
            
        elif tensor == "phi" and l>0:
            return ([i,k,l,tensor],self.update_phi(i,k,l))
        
        elif tensor == "c" :
            return ([i,k,l,tensor],self.update_c(i,k,l))
        
        else : 
            return ([],None)

    # MAPPER C - examplar based format - output the clusters assignments 
    def mapper_C(self,args):
        j,k,l = args
        for l in range(self.L):
            temp = np.add(self.A[l, :, j:k],self.R[l, :, j:k])
            self.E[l,j:k]  = np.argmax(temp, axis=1)
        
        return
            
    # Appelle l'iterateur à chaque fois sans stocker     
    def ChunkIterator(self, nb_chunk, node_format):
        
        ''' Chunk generator for the mapper function. Data is yiel depending 
        on if it should be exemplar or nodes-based as stated in the article. '''
        
        N = self.N
        chunk_size = ceil(N/nb_chunk)
        previous_index = 0
        
        if node_format==True:
            #loop on the tensor node_format
            for tensor in ['alpha','phi', 'c']:
                for l in range(self.L):
                    for i in range(chunk_size, N+chunk_size, chunk_size):
                        i = min(N,i)
                        yield [previous_index, i, l, tensor]
                        previous_index=i
                    previous_index = 0
        
        elif node_format==False:
            for tensor in ['rho', 'tau']:
                for l in range(self.L):
                    for i in range(chunk_size, N+chunk_size, chunk_size):
                        i = min(N,i)
                        yield [previous_index, i, l, tensor]
                        previous_index=i
                    previous_index = 0
                    
        else:
                chunk_size = ceil(N/nb_chunk)
                previous_index = 0
                for l in range(self.L):
                    for i in range(chunk_size, N, chunk_size):
                        yield [(previous_index, i-1), l]
                        previous_index=i
                
                
    def partitionA(self, mapped_values):

        for key, value in mapped_values:
            
            if key == []:
                continue
            i,k,l,tensor = key
        
                
            if tensor == 'alpha':
                self.A[l,:,i:k] = value.T.copy()
            elif tensor == 'phi' and l>0:
                self.PHI[l-1,i:k] = value.copy()
                
            elif tensor == 'tau' and l<self.L-1:
                self.TAU[l+1,i:k] = value.copy()
    
            elif tensor == 'c':
                self.C[l,i:k] = value.copy()
            elif tensor == 'rho':
                self.R[l,i:k,:] = value.T.copy()
            else : 
                print('Wrong Tensor key')
                
############################################### TRAIN-FIT ##################################@#######################

    # changer dans le chunk iterator : faire 2 cas node (True) ou exemplar (False) - pour eviter la boucle sur les tensors 
    def mapreduce_training(self, nb_iteration):
        
        nb_chunk = self.n_jobs
        for iteration in range(nb_iteration):
            
            map_responsesA = self.pool.map(self.mapper_A,self.ChunkIterator(nb_chunk, False))
            self.partitionA(map_responsesA)
            map_responsesB = self.pool.map(self.mapper_B,self.ChunkIterator(nb_chunk, True))
            self.partitionA(map_responsesB)

        for l in range(self.L):
            temp = np.add(self.A[l, :, :],self.R[l, :, :])
            self.E[l,:]  = np.argmax(temp, axis=1)
            print(len(Counter(self.E[l,:] ).keys()))
    
    # Pour lancer le mapreduce training, sans que ça renvoie rien 
    def fit(self, nb_iteration):
            self.mapreduce_training(nb_iteration)
            
