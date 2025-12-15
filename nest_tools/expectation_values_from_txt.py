import json
import os
import numpy as np

from scipy.constants import constants

from ase.io import read

def get_xyz(data_folder, Temp = None):
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    xyz_file = os.path.join(data_folder, 'Test.traj.extxyz')
                            
    if Temp ==None:
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
    #P = P.T[0]
    ids = np.argmax(P,axis=0)
    new_ats = []
    for i, val in enumerate(ids):
        new_ats.append([read(xyz_file,val),Temp[i]])
    return new_ats


  







