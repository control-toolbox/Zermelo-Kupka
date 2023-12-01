import numpy as np

# Function to computes self-intersections 
# of a curve in a 2-dimensional space
def get_self_intersections(curve):
    n = curve[:,0].size
    intersections = []
    for i in range(n-3):
        A = curve[i,:]
        B = curve[i+1,:]
        for j in range(i+2,n-1):
            C = curve[j,:]
            D = curve[j+1,:]
            # Matrice M : M z = b
            m11 = B[0] - A[0]
            m12 = C[0] - D[0]
            m21 = B[1] - A[1]
            m22 = C[1] - D[1]
            det = m11*m22-m12*m21
            if(np.abs(det)>1e-8):
                b1 = C[0] - A[0]
                b2 = C[1] - A[1]
                la = (m22*b1-m12*b2)/det
                mu = (m11*b2-m21*b1)/det
                if(la>=0. and la<=1. and mu>=0. and mu<=1.):
                    xx = {'i': i, 'j': j, \
                          'x': np.array(A + la * (B-A)), \
                          'la': la, 'mu': mu}
                    intersections.append(xx)
    return intersections