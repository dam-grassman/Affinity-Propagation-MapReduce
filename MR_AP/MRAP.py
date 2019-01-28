# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:22:39 2018

@author: damie
"""
from sklearn.utils import check_array
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import affinity_propagation


import multiprocessing
import numpy as np
import itertools
import warnings


def mappable_function(kwargs):
    """mappable wrapper to unpack kwargs and pass them to f"""
    
    def mapping_function(S, original_indices, preference, convergence_iter, max_iter,\
                     damping, copy, verbose,\
                     return_n_iter):
    
        #print("preference mapping", preference)
        cluster_centers_indices, labels = affinity_propagation(S, preference, convergence_iter, max_iter,\
                                                               damping, copy, verbose, return_n_iter)
        
        labels = [original_indices[cluster_centers_indices[i]] for i in labels]
        output = list(zip(original_indices, labels))
    
        return output

    return mapping_function(**kwargs)
    

class MapReduceAffinityPropagation(BaseEstimator, ClusterMixin):
    
    def __init__(self, num_workers=1, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):
        
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.pool = multiprocessing.Pool(num_workers)

        self.num_workers = num_workers
        self.nb_exemples = None

        
#    def mapping_function(self,batch, preference=None, convergence_iter=15, max_iter=200,
#                             damping=0.5, copy=True, verbose=False,
#                             return_n_iter=False):

#        S, original_indices = batch
#        cluster_centers_indices, labels = affinity_propagation(S, preference, convergence_iter, max_iter,\
#                                                               damping, copy, verbose, return_n_iter)
#
#        labels = [original_indices[cluster_centers_indices[labels[i]]] for i in labels]
#        output = list(zip(original_indices, labels))

#        return output
    
    @staticmethod
    def mappable_function(kwargs):
        """mappable wrapper to unpack kwargs and pass them to f"""
    
        def mapping_function(S, original_indices, preference, convergence_iter, max_iter,\
                         damping, copy, verbose,\
                         return_n_iter):
        
            #print("preference mapping", preference)
            cluster_centers_indices, labels = affinity_propagation(S, preference, convergence_iter, max_iter,\
                                                                   damping, copy, verbose, return_n_iter)
            
            labels = [original_indices[cluster_centers_indices[i]] for i in labels]
            output = list(zip(original_indices, labels))
        
            return output
    
        return mapping_function(**kwargs)
    
    
    def partitionA(self, mapped_values):

        labels = np.zeros(self.nb_exemples, 'int')
        
        i=0
        for key, value in mapped_values:
            labels[key] = int(value)
            i+=1
        #print('nb',i)
        return labels

    def reducerA(self, labels,  threshold = 0.5):        

        #if self.preference is None:
        #    preference = np.median(self.affinity_matrix_)
        #    print('None',preference)
            
        #else : 
        #    preference = self.preference
        #    print(preference, np.median(self.affinity_matrix_))
        
        preference = np.median(self.affinity_matrix_)
        #print(preference, preference*threshold)
        cluster_center_index = {}
        clusters_nb = 0
        
        #print(np.unique(labels))
        
        for center in np.unique(labels) :
            is_new_cluster = True
            
            for other_center in cluster_center_index.keys() : 
                
                #print(center,other_center, self.affinity_matrix_[center,other_center])
                if self.affinity_matrix_[center,other_center] >= preference*threshold: 
                    
                    cluster_center_index[center] = cluster_center_index[other_center]
                    is_new_cluster = False
                    break

            if is_new_cluster :
                cluster_center_index[center] = clusters_nb
                clusters_nb +=1
                        
        for index_point, center in enumerate(labels) :
            labels[index_point] = cluster_center_index[center]

        return labels, cluster_center_index

    def reducerB(self, X, dic_centers_, labels): 

        # inverse dictionnary of centers
        inv_dic = {}
        for center in dic_centers_ : 
            id_ = dic_centers_[center]
            l_center = inv_dic.get(id_, [])
            l_center.append(int(center))
            inv_dic[id_] = l_center

        #cluster_centers_indices = np.zeros(len(inv_dic.keys()),'int')
        cluster_centers_indices = []
        
        # compute average centroids :
        for cluster in sorted(inv_dic.keys()):

            if len(inv_dic[cluster]) ==1 : 
                cluster_centers_indices.append(X[inv_dic[cluster][0]])
            else :
                cluster_centers_indices.append(np.mean(X[inv_dic[cluster]], axis=0))
                
        return cluster_centers_indices, labels
    
    def ChunkIterator(self,S, nb_chunk, kargs):
        
        indices = np.arange(self.nb_exemples)
        np.random.shuffle(indices)
        
        chunksize = int(np.ceil(S.shape[0]/nb_chunk))
        
        #print(nb_chunk)
        for i in range(nb_chunk):
            chunk = indices[chunksize*i:chunksize*(i+1)]
            kargs['S'] = S[chunk][:,chunk]
            kargs['original_indices'] =  chunk
            #print(chunk[0])
            yield kargs.copy()
            
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def mapReduce_training(self, data, affinity_matrix, **kargs):

        #def lambda_mapping_function(batch):
        #    return self.mapping_function(batch, **kargs)
        #  Mapping
        map_responses = self.pool.map(self.mappable_function, self.ChunkIterator(affinity_matrix, self.num_workers, kargs)) 
        # Partionning for Reducer A
        #print(len(map_responses[0]),map_responses[0][0])
        #print('******************************')
        #print(len(map_responses[1]),map_responses[1][1])
        partitioned_data = self.partitionA(itertools.chain(*map_responses)) 
        # Reducer A
        labels, cluster_center_index = self.reducerA(partitioned_data) 
        #print(cluster_center_index)
        # Reducer B
        cluster_centers_indices, labels = self.reducerB(data, cluster_center_index, labels)  
        #print(cluster_centers_indices, len(labels), data.shape)
        #labels, dic_centers_ = self.pool.map(reducerA, *(partitioned_data, self.S))
        #cluster_centers_indices, labels = self.pool.map(reducerB, *[dic_centers_, labels])
        #print(np.unique(labels))
        return np.array(cluster_centers_indices), labels
    
    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def fit(self, X, y=None):

        X = check_array(X, accept_sparse='csr')
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))


        self.nb_exemples = self.affinity_matrix_.shape[0]
        
        self.cluster_centers_, self.labels_ = \
            self.mapReduce_training(data = X, affinity_matrix = self.affinity_matrix_,\
                preference = self.preference, max_iter=self.max_iter,\
                convergence_iter=self.convergence_iter, damping=self.damping,\
                copy=self.copy, verbose=self.verbose, return_n_iter=False)

        #if self.affinity != "precomputed":
        #    self.cluster_centers_ = X[self.cluster_centers_indices_].copy()
        
        self.cluster_centers_indices_ = np.zeros(len(self.cluster_centers_))
            
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_indices_")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Predict method is not supported when "
                             "affinity='precomputed'.")

        if self.cluster_centers_.size > 0:
            return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn("This model does not have any cluster centers "
                          "because affinity propagation did not converge. "
                          "Labeling every sample as '-1'.", ConvergenceWarning)
            return np.array([-1] * X.shape[0])