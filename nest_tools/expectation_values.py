import numpy as np

from ase.io import read
from scipy import constants
from scipy.spatial import cKDTree
from copy import deepcopy


def get_neigh(path, walker=200, ns_steps=4000, Temp=300):
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

    def sigmoid(x):
        return 1 / (1 + np.exp(20 * (x - 2.8)))

    atoms = read(path, ":")
    K = walker / (walker + 1)

    ref = deepcopy(atoms[0][0:8]) * [3, 3, 1]
    ref = ref.get_positions() - atoms[0].cell[0] - atoms[0].cell[1]
    tree = cKDTree(ref)

    vectors = []
    E = []
    w = []
    for i, at in enumerate(atoms[:ns_steps]):
        at.wrap()
        neigh = tree.query_ball_point(at[8].position, 5, return_length=True)
        if neigh > 0:
            d = tree.query(at[8].position, neigh)[0]
            vectors.append(np.sum(sigmoid(d)))
        else:
            vectors.append(0)

        w.append(K ** (i + 1) - K ** (i + 2))
        E.append(at.info["NS_energy"])

    vectors = np.array(vectors)
    w = np.log(w)
    E = np.array(E)
    E -= np.min(E)

    log_p = w - E / (constants.k / constants.e * Temp)
    log_p -= max(log_p)
    P = np.exp(log_p)

    return P / np.sum(P), vectors


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
    print(vectors)
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
