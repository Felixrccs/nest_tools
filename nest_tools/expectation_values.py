from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from scipy.constants import constants
from sage_lib.partition.Partition import Partition

from ase.io import iread, read

from .utils import *


def get_xyz(data_folder: str, Temp: Optional[ArrayLike] = None) -> list:
    """Return the most probable atomic structure at each temperature from a nested sampling run.

    For each temperature in ``Temp``, the Boltzmann probability of every
    sample is computed from the NS iteration indices and energies stored in
    ``Test.NS_samples``.  The sample with the highest probability is then
    looked up in the corresponding ``Test.traj.extxyz`` trajectory file and
    returned together with its temperature.

    Parameters
    ----------
    data_folder : str
        Path to the directory that contains ``Test.NS_samples`` and
        ``Test.traj.extxyz``.
    Temp : array-like, optional
        Temperatures (in K) at which to evaluate the probability.

    Returns
    -------
    list of [ase.io.iread generator, float]
        Each element is a two-item list ``[atoms_generator, T]`` where
        *atoms_generator* is the lazy ASE reader for the most-probable
        structure at temperature *T*.
    """
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


def subselect(data_folder: str, Temp: Optional[ArrayLike] = None) -> list:
    """Select one representative atomic structure per temperature, avoiding duplicates.

    The Boltzmann probability of each NS sample is computed for every
    temperature in ``Temp``.  For each temperature the samples are ranked by
    probability and the highest-ranked structure that has not already been
    selected for a previous temperature is appended to the result list.  This
    ensures that the returned structures span configuration space rather than
    repeatedly returning the same dominant sample.

    Parameters
    ----------
    data_folder : str
        Path to the directory containing ``Test.NS_samples`` and
        ``Test.traj.extxyz``.
    Temp : list or numpy.ndarray, optional
        Temperatures (in K) to consider.  If *None* or not a list/array, a
        default grid spanning 100–4100 K is used.

    Returns
    -------
    list of ase.Atoms
        One ``Atoms`` object per temperature in ``Temp``, chosen to maximise
        coverage of distinct configurations.
    """
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    xyz_file = os.path.join(data_folder, 'Test.traj.extxyz')
                            
    if type(Temp) not in [list,np.array]:
        Temp = np.concat([np.arange(start=100,stop=3100,step=15), np.arange(start=3100,stop=4100,step=50)])
        print(Temp)
    
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
        new_ats.append(read(xyz_file,val))
    return new_ats






def get_probabiliy(
    data_folder: str,
    Temp: Optional[ArrayLike] = None,
    iter: int = 1,
    observable: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Boltzmann weights and a thermodynamic observable from NS output files.

    Reads the nested-sampling samples file ``Test.NS_samples`` and an
    associated ``sage_lib`` ``Partition`` database to compute the Boltzmann
    probability matrix *P* at each requested temperature.  The function then
    evaluates one of several pre-defined observables (RDF, structure factor,
    graph topology, …) or falls back to returning the raw energies.

    Parameters
    ----------
    data_folder : str
        Path to the directory containing ``Test.NS_samples`` and the ``db``
        sub-directory used by ``sage_lib.partition.Partition``.
    Temp : array-like, optional
        Temperatures (in K) at which to evaluate the probability.  Defaults
        to ``np.arange(100, 2500, 50)`` if not provided.
    iter : int, optional
        Stride for sub-sampling the NS samples (every *iter*-th row is used).
        Default is 1 (no sub-sampling).
    observable : {None, 'rdf', 'rdf_full', 'sf', 'subG', 'graph', 'atoms'}
        Which observable to compute:

        * ``None``      – raw energies from the samples file.
        * ``'rdf'``     – radial distribution function (partial).
        * ``'rdf_full'``– full radial distribution function.
        * ``'sf'``      – structure factor.
        * ``'subG'``    – subgraph / motif count.
        * ``'graph'``   – CuO graph descriptor.
        * ``'atoms'``   – per-atom properties.

    Returns
    -------
    Temp : numpy.ndarray
        Temperature grid used for the calculation.
    P : numpy.ndarray, shape (n_samples, n_temps)
        Normalised Boltzmann probability of each sample at each temperature.
    n_observable : numpy.ndarray
        The evaluated observable array corresponding to *Temp*.
    """
    data_file = os.path.join(data_folder, 'Test.NS_samples')
    db_file = os.path.join(data_folder, 'db')

    p = Partition(
        storage="hybrid",
        local_root=db_file,
        access="ro",    # "ro" = read-only, "rw" = read/write
    )
    
    if type(Temp) not in [list,np.array]:
        Temp = np.arange(start=100,stop=2500,step=50)

    def _get_P_from_txt(data_file: str, iter: int) -> np.ndarray:
        """Read ``data_file`` and return the Boltzmann probability matrix.

        Parameters
        ----------
        data_file : str
            Path to the ``Test.NS_samples`` file.
        iter : int
            Row stride for sub-sampling (every *iter*-th sample is kept).

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_temps)
            Normalised Boltzmann probabilities.
        """
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


  







