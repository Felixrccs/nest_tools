import numpy as np

from ase.io import read
from ase.neighborlist import neighbor_list

from scipy import constants
from scipy.spatial import cKDTree
from copy import deepcopy



def get_neigh(path, walker=200, ns_steps=None, Temp=300):
    """builds a smooth CN for the central atom

    :param path: Path to nested sampling xyz
    :type path: string
    :param walker: count of used walkers
    :type walker: int, optional
    :param ns_steps: amount of ns steps included, defaults to 3000
    :type ns_steps: int, optional
    :param Temp: Temperature, defaults to 300
    :type Temp: int, optional
    """
    Temp = np.array(Temp)

    def sigmoid(x):
        return 1 / (1 + np.exp(20 * (x - 3.8)))

    if type(ns_steps) == int:
        atoms = read(path, f":{ns_steps}")
    else:
        atoms = read(path, ":")
    
    K = walker / (walker + 1)


    vectors = np.zeros((len(atoms),25))
    E = []
    w = []
    for i, at in enumerate(atoms):
        idx, d = neighbor_list('id',at, 4.5)

        u_id, counts = np.unique(idx,return_counts=True)
        per_atom = np.array_split(d,np.cumsum(counts))
        per_atom.pop(-1)
        CN_s = np.zeros_like(at.get_tags())
        for j,val in enumerate(per_atom):
            CN_s[u_id[j]]= int(np.sum(sigmoid(val)))

        bins = np.unique(CN_s[32:], return_counts=True)
        vectors[i][bins[0]] = bins[1]
        w.append(K ** (i + 1) - K ** (i + 2))
        E.append(at.info["NS_energy"])

    w = np.log(w)
    E = np.array(E)
    E -= np.min(E)

    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T
    print(i)
    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T
    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    return P,vectors

    
    #log_p = w - E / (constants.k / constants.e * Temp)
    #log_p -= max(log_p)
    #P = np.exp(log_p)

    #return P / np.sum(P), vectors


def get_site(path, walker, ns_steps=4000, Temp=[300], supercell=4):
    """funcion to calculate population of absorbate sites

    :param path: Path to nested sampling xyz
    :type path: string
    :param walker: count of used walkers
    :type walker: int, optional
    :param ns_steps: amount of ns steps included, defaults to 3000
    :type ns_steps: int, optional
    :param Temp: Temperature, defaults to 300
    :type Temp: array, optional
    """
    Temp = np.array(Temp)
    atoms = read(path, ":")
    K = walker / (walker + 1)

    vectors = []
    E = []
    w = []
    empty = np.arange(0,supercell*supercell)
    for i, at in enumerate(atoms[:ns_steps]):
        at.set_scaled_positions(at.get_scaled_positions() + np.array([0.25, 0.25, 0]))
        at.wrap()
        tmp = at[-1].scaled_position[0:2]
        tmp = np.abs(np.round((tmp - 1/(2*supercell))*supercell ))
        tmp[1]*=supercell
        vectors.append(np.where(empty == np.sum(tmp),1,0))

        w.append(K ** (i + 1) - K ** (i + 2))
        E.append(at.info["NS_energy"])

    vectors = np.array(vectors)
    w = np.log(w)
    E = np.array(E)
    E -= np.min(E)
    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T

    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T
    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    return np.matmul(P.T,vectors)



def get_O_distance(path, walker, Temp=[300], supercell=4):
    """funcion to calculate distance between atoms

    :param path: Path to nested sampling xyz
    :type path: string
    :param walker: count of used walkers
    :type walker: int, optional
    :param ns_steps: amount of ns steps included, defaults to 3000
    :type ns_steps: int, optional
    :param Temp: Temperature, defaults to 300
    :type Temp: array, optional
    """
    Temp = np.array(Temp)
    if type(path) == str:
        atoms = read(path, ":")
    else:
        atoms = path
    K = walker / (walker + 1)

    vectors = []
    E = []
    w = []
    for i, at in enumerate(atoms):
        vectors.append(at.get_distance(-1,-2, mic=True))
        w.append(K ** (i + 1) - K ** (i + 2))
        E.append(at.info["NS_energy"])

    vectors = np.array(vectors)
    w = np.log(w)
    E = np.array(E)
    E -= np.min(E)
    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T

    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T
    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    return P.T,vectors









def get_weight(path, walker, Temp=[50]):
    """funcion to calculate population of absorbate sites

    :param path: Path to nested sampling xyz
    :type path: string
    :param walker: count of used walkers
    :type walker: int, optional
    :param ns_steps: amount of ns steps included, defaults to 3000
    :type ns_steps: int, optional
    :param Temp: Temperature, defaults to 300
    :type Temp: array, optional
    """
    Temp = np.array(Temp)
    atoms = read(path, ":")
    K = walker / (walker + 1)

    vectors = []
    E = []
    w = []
    for i, at in enumerate(atoms):
        at.wrap()
        w.append(K ** (i + 1) - K ** (i + 2))
        E.append(at.info["NS_energy"])

    vectors = np.array(vectors)
    w = np.log(w)
    E = np.array(E)
    E -= np.min(E)
    w_mat = np.broadcast_to(w[:, None], (len(E), len(Temp)))
    E_mat = np.broadcast_to(E[:, None], (len(E), len(Temp)))
    T_mat = np.broadcast_to(Temp[:, None], (len(Temp), len(E))).T

    log_p = w_mat - E_mat / (constants.k / constants.e * T_mat)
    log_p -= np.broadcast_to(np.max(log_p, axis=0)[:, None], (len(Temp), len(E))).T
    P = np.exp(log_p)
    P /= np.broadcast_to(np.sum(P, axis=0)[:, None], (len(Temp), len(E))).T
    P = P.T[0]
    print(np.max(P)/P[-1])
    i = 0.01
    while np.sum(np.where(P > i,P,0))<0.9:
        i /= 2
        print(i)
    print(np.sum(np.where(P > i,P,0)))
    ids = np.where(P > i)[0]
    print(ids)
    return atoms[min(ids):(max(ids)+1)]
