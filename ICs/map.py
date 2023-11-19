
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix

def map(mesh, xlon, ylat, fdat):

#-- remap unstructured [x,y,f] onto mesh cells

    ppos = np.zeros(
        (mesh.cell.size, 3), dtype=np.float64)
    ppos[:, 0] = mesh.cell.xpos
    ppos[:, 1] = mesh.cell.ypos
    ppos[:, 2] = mesh.cell.zpos

    tree = spatial.cKDTree(ppos, leafsize=8)

    qpos = np.zeros(
        (xlon.shape[ 0], 3), dtype=np.float64)
    qpos[:, 0] = mesh.rsph * np.cos(xlon) * \
                             np.cos(ylat)
    qpos[:, 1] = mesh.rsph * np.sin(xlon) * \
                             np.cos(ylat)
    qpos[:, 2] = mesh.rsph * np.sin(ylat)
    
    try:  # ridiculous argument renaming...
        __, near = tree.query(qpos, n_jobs=-1)
    except:
        __, near = tree.query(qpos, workers=-1)

    cols = np.arange(0, near.size)
    vals = np.ones(near.size, dtype=np.int8)

    smat = csr_matrix((vals, (near, cols)), 
        shape=(mesh.cell.size, near.size), dtype=np.int8)
        
    nmap = np.asarray(smat.sum(axis=1), dtype=np.int32)
    
    fdat = np.reshape(fdat, (fdat.size, 1))
    
    fmap = (smat * fdat) / np.maximum(1, nmap)
    
    fmap = np.reshape(fmap, (fmap.size))

    return fmap
    
