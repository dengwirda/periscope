
import os
import sys
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix

HERE = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(HERE, "ext"))

#-- simple interfaces to Akima's interpolation schemes
from ext.akima import interp1d, interp2d

def flatten(alat, flat):
#-- geodetic => geocentric mapping for (mean) spheroid
    return np.arctan(
        (1. - flat) ** 2 * np.tan(alat))


def idw_remap(ppos, qpos, halo, dpow=4):

#-- return inverse distance weighted interpolation matrix

    tree = spatial.cKDTree(ppos, leafsize=32)

    try:  # scipy renaming
        dist, near = \
            tree.query(qpos, k=halo, n_jobs=-1)
    except:
        dist, near = \
            tree.query(qpos, k=halo, workers=-1)

    scal = dist / np.mean(dist)  # careful w. precision

    tiny = 1.E-12 * np.mean(scal)

    wght = (1. / np.maximum(tiny, scal)) ** dpow
   
    dist = np.mean(dist, axis=+1)

    wsum = np.maximum(tiny, np.sum(wght, axis=1))

    nbse = ppos.shape[0]
    npts = qpos.shape[0]

    ivec = []; jvec = []; xvec = []
    for next in range(halo):
        ivec.append(np.arange(npts))
        jvec.append(near[:, next])
        xvec.append(wght[:, next] / wsum)

    ivec = np.concatenate(ivec)
    jvec = np.concatenate(jvec)
    xvec = np.concatenate(xvec)

    return csr_matrix(
        (xvec, (ivec, jvec)), shape=(npts, nbse)), dist


def find_cell(mesh, xlon, ylat):

#-- nearest cells to unstructured [x,y] points

    ppos = np.zeros(
        (mesh.cell.size, 3), dtype=np.float64)
    ppos[:, 0] = mesh.cell.xpos
    ppos[:, 1] = mesh.cell.ypos
    ppos[:, 2] = mesh.cell.zpos

    tree = spatial.cKDTree(ppos, leafsize=32)

    qpos = np.zeros(
        (xlon.shape[ 0], 3), dtype=np.float64)
    qpos[:, 0] = mesh.rsph * np.cos(xlon) * \
                             np.cos(ylat)
    qpos[:, 1] = mesh.rsph * np.sin(xlon) * \
                             np.cos(ylat)
    qpos[:, 2] = mesh.rsph * np.sin(ylat)
    
    try:  # scipy renaming
        __, near = tree.query(qpos, n_jobs=-1)
    except:
        __, near = tree.query(qpos, workers=-1)
        
    return near
    
    
def find_edge(mesh, xlon, ylat):

#-- nearest edges to unstructured [x,y] points

    ppos = np.zeros(
        (mesh.edge.size, 3), dtype=np.float64)
    ppos[:, 0] = mesh.edge.xpos
    ppos[:, 1] = mesh.edge.ypos
    ppos[:, 2] = mesh.edge.zpos

    tree = spatial.cKDTree(ppos, leafsize=32)

    qpos = np.zeros(
        (xlon.shape[ 0], 3), dtype=np.float64)
    qpos[:, 0] = mesh.rsph * np.cos(xlon) * \
                             np.cos(ylat)
    qpos[:, 1] = mesh.rsph * np.sin(xlon) * \
                             np.cos(ylat)
    qpos[:, 2] = mesh.rsph * np.sin(ylat)
    
    try:  # scipy renaming
        __, near = tree.query(qpos, n_jobs=-1)
    except:
        __, near = tree.query(qpos, workers=-1)
        
    return near
    
    
def find_dual(mesh, xlon, ylat):

#-- nearest duals to unstructured [x,y] points

    ppos = np.zeros(
        (mesh.vert.size, 3), dtype=np.float64)
    ppos[:, 0] = mesh.vert.xpos
    ppos[:, 1] = mesh.vert.ypos
    ppos[:, 2] = mesh.vert.zpos

    tree = spatial.cKDTree(ppos, leafsize=32)

    qpos = np.zeros(
        (xlon.shape[ 0], 3), dtype=np.float64)
    qpos[:, 0] = mesh.rsph * np.cos(xlon) * \
                             np.cos(ylat)
    qpos[:, 1] = mesh.rsph * np.sin(xlon) * \
                             np.cos(ylat)
    qpos[:, 2] = mesh.rsph * np.sin(ylat)
    
    try:  # scipy renaming
        __, near = tree.query(qpos, n_jobs=-1)
    except:
        __, near = tree.query(qpos, workers=-1)
        
    return near
    

def maptocell(mesh, xlon, ylat, fdat):

#-- remap unstructured [x,y,f] onto mesh cells

    near = find_cell(mesh, xlon, ylat)

    cols = np.arange(0, near.size)
    vals = np.ones(near.size, dtype=np.int8)

    smat = csr_matrix((vals, (near, cols)), 
        shape=(mesh.cell.size, near.size), dtype=np.int8)
        
    nmap = np.asarray(smat.sum(axis=1), dtype=np.int32)
    
    fdat = np.reshape(fdat, (fdat.size, 1))
    
    fmap = (smat * fdat) / np.maximum(1, nmap)
    
    fmap = np.reshape(fmap, (fmap.size))

    return fmap
    
    
def maptoedge(mesh, xlon, ylat, fdat):

#-- remap unstructured [x,y,f] onto mesh edges

    near = find_edge(mesh, xlon, ylat)

    cols = np.arange(0, near.size)
    vals = np.ones(near.size, dtype=np.int8)

    smat = csr_matrix((vals, (near, cols)), 
        shape=(mesh.edge.size, near.size), dtype=np.int8)
        
    nmap = np.asarray(smat.sum(axis=1), dtype=np.int32)
    
    fdat = np.reshape(fdat, (fdat.size, 1))
    
    fmap = (smat * fdat) / np.maximum(1, nmap)
    
    fmap = np.reshape(fmap, (fmap.size))

    return fmap
    
    
def maptodual(mesh, xlon, ylat, fdat):

#-- remap unstructured [x,y,f] onto mesh duals

    near = find_dual(mesh, xlon, ylat)

    cols = np.arange(0, near.size)
    vals = np.ones(near.size, dtype=np.int8)

    smat = csr_matrix((vals, (near, cols)), 
        shape=(mesh.vert.size, near.size), dtype=np.int8)
        
    nmap = np.asarray(smat.sum(axis=1), dtype=np.int32)
    
    fdat = np.reshape(fdat, (fdat.size, 1))
    
    fmap = (smat * fdat) / np.maximum(1, nmap)
    
    fmap = np.reshape(fmap, (fmap.size))

    return fmap
    
