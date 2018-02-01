import numpy as np

def gen_con(nNodes):
    '''
        Generates a structural connectivity matrix such that
        the weights follow exponential distribution
    '''
    SC = np.zeros([nNodes,nNodes])
    for i in range(nNodes):
        for j in range(i, nNodes):
            if(i ==j):
                SC[i,j] = 0
            else:      
                SC[i,j] = SC[j,i] = np.random.exponential(0.1)
    return SC

def comp_proj_mat(src_locs,snsr_locs):
    '''
        Compute the projection matrix from sources to sensors
    '''
    nNodes = np.size(src_locs,0)
    nSnsrs = np.size(snsr_locs,0)
    proj_mat = np.zeros([nSnsrs, nNodes])
    for i in range(nSnsrs):
        t = np.linalg.norm(snsr_locs[i,:] - src_locs,ord=None,axis=1)     
        proj_mat[i,:] = 1.0 / t**2
    return proj_mat
