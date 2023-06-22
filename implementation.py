import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd

class SinkhornKnopp:
    def __init__(self,c,X,Y,mu,nu,eps):
        """c is the cost function of our problem
        X and Y are subsets of the measurable spaces we are working on, X being the departure subset.
        Note that both X and Y are subset of $\mathbb{R}^{2}$ and are the same shape as X = [(x_1,y_1),(x_2,y_2),...,(x_N1,y_N1)] (same for Y)

        mu and nu are respectively are measures on X and Y. 
        For example, mu is the same shape as mu = [mu_1, mu_2, ..., mu_N1]. For every i \in {0,...,N1-1}, mu[i] is the mass of the point X[i].
        Such that this vector mu is representing the measure $\sum_{i=0}^{N1-1} mu[i] dirac_{X[i]}(.)$. (As we are only working with discrete measures)

        Eps is a positive real number that we use for the entropy model, see : https://lchizat.github.io/ot2021orsay.html lecture 3 for a presentation of the model"""

        self.cost = c
        self.X = X
        self.N1 = len(X)
        self.Y = Y
        self.N2 = len(Y)
        self.mu = mu
        self.nu = nu
        self.eps = eps
        self.D_phi = None
        self.D_psi = None
        self.K = None


    def e(r):
        """This function is used to calculate the entropy in the paper mentionned above (lecture 3) although is doesn't have any purpose here."""
        if r > 0:
            return r*(math.log(r)-1)
        if r==0:
            return 0
        return np.inf
    
    def sk_measures_init(self,k_max = 50):
        """This function takes for entrance a maximal number of iterations and calculates D_phi and D_psi, matrixes such as diag(D_phi)*K*diag(D_psi) gives us how the mass of the points in X is distributed to the points in Y
        K a matrix that we find after resolving the first order condition on $\gamma_{xy}$ (cf. the " The Entropic Optimal Transport" part of the lecture 3)"""
        K = np.zeros((self.N1,self.N2))
        for i in range(self.N1):
            for j in range(self.N2):
                K[i][j] = math.exp((-self.cost(self.X[i],self.Y[j]))/self.eps)

        D_phi = np.ones((self.N1,1))
        D_psi = np.ones((self.N2,1))

        for k in range(k_max):
            D_phi = mat_div(self.mu,np.matmul(K,D_psi))
            D_psi = mat_div(self.nu,np.matmul(np.transpose(K),D_phi))
        
        self.D_phi = D_phi
        self.D_psi = D_psi
        self.K = K
    
    def sk_measures(self,k_max = 50):

        if self.D_phi is None:
            self.sk_measures_init(k_max)

        mat1 = np.zeros((self.N1,self.N1))
        mat2 = np.zeros((self.N2,self.N2))
    
        for k in range(self.N1):
            mat1[k][k]=self.D_phi[k][0]
        for k in range(self.N2):
            mat2[k][k]=self.D_psi[k][0]
        res = np.matmul(mat1,np.matmul(self.K,mat2))

        for i in range(self.N1):
            ind_max = 0
            max = res[i][0]
            for j in range(self.N2):
                if res[i][j]>max:
                    max = res[i][j]
                    ind_max = j
            plt.plot([self.X[i][0],self.Y[ind_max][0]],[self.X[i][1],self.Y[ind_max][1]])
            plt.plot([self.X[i][0]],[self.X[i][1]],"o",color="black",markersize=1)
            plt.plot([self.Y[ind_max][0]],[self.Y[ind_max][1]],"o",color="red",markersize=1)
            #plt.plot([self.X[i][0],self.Y[ind_max][0]],[self.X[i][1],self.Y[ind_max][1]],label=f"DLPIPDP {i} = {str(res[i][ind_max])[0:5]}")
        #plt.legend(loc="upper left")
        plt.show()
        

def mat_div(A,B):
    """Division (element-wise) between two matrix"""
    n = len(A)
    m = len(A[0])
    C = [[0 for k in range(m)]for i in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j]/B[i][j]
    return C