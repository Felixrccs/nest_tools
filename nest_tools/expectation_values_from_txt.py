import json
import os
import numpy as np

from scipy.constants import constants
from sage_lib.partition.Partition import Partition

from ase.io import iread

from .utils import *


def get_xyz(data_folder, Temp = None):
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    xyz_file = os.path.join(data_folder, 'Test.traj.extxyz')
                            
    #if np.anTemp == None:
    #    Temp = np.arange(start=100,stop=2500,step=50)
    
    with open(data_file, 'r') as file:
        first_line = file.readline().strip()
        params = json.loads(first_line[2:])

    data = np.loadtxt(data_file,skiprows=1).T

    K = params["n_walkers"]/(params[ "n_walkers"]+1)

    E = data[1]
    w = np.log(K ** (data[0] + 1) - K ** (data[0] + 2))
    E -= np.min(E)
    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T
    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T

    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    #P = P.T[0]
    ids = np.argmax(P,axis=0)
    new_ats = []
    for i, val in enumerate(ids):
        new_ats.append([iread(xyz_file,val),Temp[i]])
    return new_ats


def subselect(data_folder, Temp = None):
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    xyz_file = os.path.join(data_folder, 'Test.traj.extxyz')
                            
    if type(Temp) not in [list,np.array]:
        Temp = np.arange(start=100,stop=2100,step=10)
    
    with open(data_file, 'r') as file:
        first_line = file.readline().strip()
        params = json.loads(first_line[2:])

    data = np.loadtxt(data_file,skiprows=1).T

    K = params["n_walkers"]/(params[ "n_walkers"]+1)

    E = data[1]
    w = np.log(K ** (data[0] + 1) - K ** (data[0] + 2))
    E -= np.min(E)
    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T
    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T

    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    #P = P.T[0]
    new_ids = []
    for val in P.T:
        ids = np.argsort(val)[::-1]
        count = 0
        while ids[count] in new_ids:
            count+=1
        new_ids.append(ids[count])

    new_ats = []
    for val in tqdm(new_ids):
        new_ats.append(*iread(xyz_file,val))
    return new_ats


def get_nn(data_folder, Temp = None):
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    xyz_file = os.path.join(data_folder, 'Test.traj.extxyz')

    atoms = read(xyz_file,':')

    nn = get_connectivity(atoms)
                            
    if type(Temp) not in [list,np.array]:
        Temp = np.arange(start=100,stop=2500,step=50)
    
    with open(data_file, 'r') as file:
        first_line = file.readline().strip()
        params = json.loads(first_line[2:])

    data = np.loadtxt(data_file,skiprows=1).T

    K = params["n_walkers"]/(params[ "n_walkers"]+1)

    E = data[1]
    w = np.log(K ** (data[0] + 1) - K ** (data[0] + 2))
    E -= np.min(E)
    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T
    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T

    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    return P, nn




def get_probabiliy(data_folder, Temp = None, iter = 1, observable = None):
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    db_file = os.path.join(data_folder, 'db')

    p = Partition(
        storage="hybrid",
        local_root=db_file,
        access="ro",    # "ro" = read-only, "rw" = read/write
    )
    
    if type(Temp) not in [list,np.array]:
        Temp = np.arange(start=100,stop=2500,step=50)

    def _get_P_from_txt(data_file, iter):
        with open(data_file, 'r') as file:
            first_line = file.readline().strip()
            params = json.loads(first_line[2:])

        data = np.loadtxt(data_file,skiprows=1)[::iter].T

        K = params["n_walkers"]/(params[ "n_walkers"]+1)

        E = data[1]
        w = np.log(K ** (data[0] + 1) - K ** (data[0] + 2))
        E -= np.min(E)
        w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
        E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
        T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T
        log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
        log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T

        P = np.exp(log_p)
        P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
        return P
    if iter >1:
        p = p[::iter]
    
    P = _get_P_from_txt(data_file=data_file,iter=iter)
    if observable == 'rdf':
        n_observable = get_rdf(p)
    if observable == 'rdf_full':
        n_observable = get_rdf_full(p)
    if observable == 'sf':
        n_observable = get_sf(p)
    if observable == 'subG':
        n_observable = get_graph_count(p)
    if observable == 'graph':
        n_observable = get_graph_CuO(p)
    if observable == 'atoms':
        n_observable = get_atoms(p)
    if observable == None:
        n_observable = np.loadtxt(data_file,skiprows=1)[::iter].T[1]
    return Temp, P, n_observable


  







