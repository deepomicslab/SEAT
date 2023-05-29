# -*- coding: utf-8 -*-
"""
    src.HE
    ~~~~~~~~~~~

    @Copyright: (c) 2022-07 by Lingxi Chen (chanlingxi@gmail.com).
    @License: LICENSE_NAME, see LICENSE for more details.
"""

import torch
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class HierachicalEmbedding(object):

    def __init__(self, n_components=2,
                 affinity='precomputed',
                 n_epochs=None,
                 learning_rate=1e-2,
                 init='spectral',
                 spectral_init_y_idx=1,
                 min_dist=0.1,
                 spread = 1,
                 device='cpu',
                 verbose=True,
                 random_state=None):
        self.n_components = n_components
        self.affinity = affinity
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.spectral_init_y_idx = spectral_init_y_idx
        self.min_dist = min_dist
        self.spread = spread
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        
        self.a, self.b = find_ab_params(self.spread, self.min_dist)
        if self.verbose:
            print('spread = {}, min_dist = {}, a = {}, b = {}'.format(self.spread, self.min_dist, self.a, self.b))
        
    def get_similarity(self, X):
        D = self.get_distance(X)
        S = 1 / (1 + self.a*(D**self.b))
        return S

    def get_distance(self, X):
        M, N = X.shape
        G = torch.mm(X, X.t())
        g = torch.diag(G)
        x = g.repeat(M, 1)
        D = x + x.t() - 2*G
        return D
    
    def replace_submatrix(self, mat, ind1, ind2, mat_replace):
        for i, index in enumerate(ind1):
            mat[index, ind2] = mat_replace[i, :]
    
    def process_affinity(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        aff_m = X
        self.aff_m_list = []
        self.aff_t_list = []

        for i in range(y.shape[1]):
            tmp_aff_m = np.zeros((self.n_sample, self.n_sample))
            labels = y[:, i]           
            for label in set(labels):
                idx = np.argwhere(labels == label)[:, 0]
                self.replace_submatrix(tmp_aff_m, idx, idx, aff_m[np.ix_(idx, idx)]) 
                
            self.aff_m_list.append(tmp_aff_m)
            self.aff_t_list.append(torch.tensor(tmp_aff_m, dtype=torch.float, device=self.device))
        
    def init_U(self):
        if self.init == 'random':
            generator = torch.Generator(device=self.device)
            if self.random_state is not None:
                generator.manual_seed(self.random_state)
            if self.aff_m_list[0].shape[0] > 50:
                k = 50
                k = self.aff_m_list[0].shape[0]
            else:
                k = self.aff_m_list[0].shape[0]

            U = torch.rand(self.n_sample, k, dtype=torch.float, 
                           device=self.device, generator=generator, requires_grad=True)
            self.inital_U = U.data.cpu().detach().numpy().copy()        
        if self.init == 'spectral':
            from sklearn.manifold import SpectralEmbedding
            self.inital_U = SpectralEmbedding(n_components=self.n_components, random_state=self.random_state,
                                              affinity='precomputed').fit_transform(self.aff_m_list[self.spectral_init_y_idx])
            U = torch.tensor(self.inital_U, dtype=torch.float, device=self.device, requires_grad=True)
        return U
    
    def lap_loss(self, X, Sw):
        return torch.sum(get_distance(X)*Sw)

    def bce_loss(self, club_S, U_S, eph=1e-4):
        return torch.sum(club_S*torch.log(torch.div(club_S, U_S+eph)+eph) + (1-club_S)*torch.log(torch.div(1-club_S, 1-U_S+eph)+eph))

    def fit_transform(self, X, y, thetas=[0.0, 0.2, 0.8]):

        self.n_sample = X.shape[0]
        self.y = y
        if y.shape[1] == 1:
            self.spectral_init_y_idx = 0
        self.thetas = thetas
        self.process_affinity(X, y)

        U = self.init_U()
        
        optimizer = torch.optim.Adam([U], lr=self.learning_rate, weight_decay=0.0001)  # append L2 penalty, 

        loss_list = []
          
        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if X.shape[0] <= 10000:
                if self.init == 'spetral':
                    self.n_epochs = 1000
                else:
                    self.n_epochs = 500
            else:
                if self.init == 'spetral':
                    self.n_epochs = 600
                else:
                    self.n_epochs = 200  
        
        
        for i in range(self.n_epochs):
            self.a = 1
            self.b = 1
            U_S = self.get_similarity(U)
    
            info = 'epoch={}\t'.format(i)
        
            loss = 0
            for j in range(0, len(self.aff_t_list)):
                aff_m = self.aff_t_list[j]
                tmp_loss = thetas[j] * self.bce_loss(aff_m, U_S)
                loss = loss + tmp_loss
                info += 'loss_{}={}\t'.format(j, tmp_loss)
            info += 'loss={}\t'.format(loss)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            info += 'update_loss={}\t'.format(current_loss)
            loss_list.append(current_loss)
           
            if i % 100 == 0 and self.verbose:
                print(info)
        if self.verbose:
            print(info)    
        
        self.U = U
        self.loss_list = loss_list
        self.embed_U = U.data.cpu().detach().numpy()[:,:self.n_components]
        return self.embed_U    

    def viz_fit(self, n_row=3, n_col=None, fig_width=5, fn=None):
        if n_col is None:
            n_col = self.y.shape[1]
        point_size = 100.0 / np.sqrt(self.inital_U.shape[0])
        plt.figure(figsize=(fig_width*n_col, fig_width*n_row))
        fig_count = 1
        
        if isinstance(self.y, np.ndarray):
            y = pd.DataFrame(self.y, columns=['col{}'.format(i) for i in range(self.y.shape[1])])
        else:
            y = self.y
        
        args = 'min_dist={},thetas={}'.format(self.min_dist, self.thetas)
        for i in range(y.shape[1]):
            plt.subplot(n_row, n_col, fig_count)
            plt.title('{} init,c={}\n{}'.format(self.init, y.columns[i], args))
            plt.scatter(self.inital_U[:, 0], self.inital_U[:, 1], c=y[y.columns[i]], 
                        s=point_size,
                        cmap=ListedColormap(sns.color_palette('Spectral', len(set(y[y.columns[i]])))))
            fig_count += 1

        for i in range(self.y.shape[1]):
            plt.subplot(n_row, n_col, fig_count)
            plt.title('HE,c={}\n{}'.format(y.columns[i], args))
            plt.scatter(self.embed_U[:, 0], self.embed_U[:, 1], 
                        s=point_size,
                        c=y[y.columns[i]], 
                        cmap=ListedColormap(sns.color_palette('Spectral', len(set(y[y.columns[i]])))))
            fig_count += 1

        plt.subplot(n_row, n_col, fig_count)
        plt.title('Training loss')
        plt.plot(self.loss_list)
        fig_count += 1 
        
        if fn is None:
            plt.show()
        else:
            plt.savefig(fn)
       

   


