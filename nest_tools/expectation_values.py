from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sage_lib.partition.Partition import Partition
from scipy import constants

from .utils import get_atoms, get_dist, get_graph_CuO, get_sf, get_z

# In-plane edge length (Å) of the 4×4 Cu surface cell used for the 2D density.
CELL_LENGTH = 14.52557132


def get_probability(
    data_folder: str,
    Temp: Optional[ArrayLike] = None,
    stride: int = 1,
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
    stride : int, optional
        Stride for sub-sampling the NS samples (every *stride*-th row is
        used). Default is 1 (no sub-sampling).
    observable : {None, 'z', 'dist', 'sf', 'graph', 'atoms', 'energy'}
        Which observable to compute:

        * ``None``       – the partition itself (a list of containers if
          *stride* > 1).
        * ``'z'``        – mean and maximum z-position of the surface layer.
        * ``'dist'``     – Cu–Cu pair distances of the surface layer.
        * ``'sf'``       – structure factor descriptor.
        * ``'graph'``    – heteroatomic CuO neighbor graphs.
        * ``'atoms'``    – ASE ``Atoms`` objects.
        * ``'energy'``   – raw energies from the samples file.

    Returns
    -------
    Temp : numpy.ndarray
        Temperature grid used for the calculation.
    P : numpy.ndarray, shape (n_samples, n_temps)
        Normalised Boltzmann probability of each sample at each temperature.
    n_observable : numpy.ndarray or list
        The evaluated observable, one entry per (sub-sampled) NS sample.

    Raises
    ------
    ValueError
        If *observable* is not one of the recognised options.
    """
    data_file = os.path.join(data_folder, "Test.NS_samples")
    db_file = os.path.join(data_folder, "db")

    p = Partition(
        storage="hybrid",
        local_root=db_file,
        access="ro",  # "ro" = read-only, "rw" = read/write
    )

    if Temp is None:
        Temp = np.arange(start=100, stop=2500, step=50)
    Temp = np.asarray(Temp)

    def _get_P_from_txt(data_file: str, stride: int) -> np.ndarray:
        """Read ``data_file`` and return the Boltzmann probability matrix.

        Parameters
        ----------
        data_file : str
            Path to the ``Test.NS_samples`` file.
        stride : int
            Row stride for sub-sampling (every *stride*-th sample is kept).

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_temps)
            Normalised Boltzmann probabilities.
        """
        with open(data_file, "r") as file:
            first_line = file.readline().strip()
            params = json.loads(first_line[2:])

        data = np.loadtxt(data_file, skiprows=1)[::stride].T

        K = params["n_walkers"] / (params["n_walkers"] + 1)

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

    if stride > 1:
        p = p[::stride]

    P = _get_P_from_txt(data_file=data_file, stride=stride)
    if observable is None:
        n_observable = p
    elif observable == "z":
        n_observable = get_z(p)
    elif observable == "dist":
        n_observable = get_dist(p)
    elif observable == "sf":
        n_observable = get_sf(p)
    elif observable == "graph":
        n_observable = get_graph_CuO(p)
    elif observable == "atoms":
        n_observable = get_atoms(p)
    elif observable == "energy":
        n_observable = np.loadtxt(data_file, skiprows=1)[::stride].T[1]
    else:
        raise ValueError(f"Unknown observable: {observable!r}")
    return Temp, P, n_observable


# Backward-compatible alias for the original (misspelled) name.
get_probabiliy = get_probability


def calculate_rdf(
    path: str, stride: int = 1, n_atoms: int = 24
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Compute the temperature-dependent 2D radial distribution function g(r).

    Pair distances are collected for every nested-sampling configuration
    (observable ``'dist'``) and histogrammed with Boltzmann probability
    weights, yielding one RDF per temperature on a grid from 2450 K down to
    100 K in steps of 50 K.

    The normalisation is two-dimensional: counts are divided by the annulus
    area ``π (r₂² − r₁²)`` of each bin and by the ideal 2D density
    ``n_atoms / CELL_LENGTH²``. Distances are binned between 1.5 and 4.5 Å
    (199 bins).

    Parameters
    ----------
    path : str
        Path to the nested-sampling run folder (containing
        ``Test.NS_samples`` and the ``db`` sub-directory).
    stride : int
        Stride for sub-sampling the NS samples (every *stride*-th row is
        used).
    n_atoms : int
        Number of atoms contributing to the pair distances; sets the ideal
        2D density and the per-atom probability weighting.

    Returns
    -------
    T : numpy.ndarray
        Temperature grid (K), descending from 2450 to 100.
    bin_centers : numpy.ndarray
        Histogram bin centers (Å), shape ``(199,)``.
    all_rdf : list[numpy.ndarray]
        One 2D RDF per temperature, each of shape ``(199,)``.
    """
    T, P, data = get_probability(
        path,
        stride=stride,
        observable="dist",
        Temp=np.arange(start=2450, stop=99, step=-50),
    )

    d_stack = np.concatenate(data)
    length = np.array([len(val) for val in data])

    _bins = np.linspace(1.5, 4.5, 200)
    bin_centers = 0.5 * (_bins[:-1] + _bins[1:])
    _shell_area = np.pi * (
        np.power(_bins[1:], 2) - np.power(_bins[:-1], 2)
    )  # 2D normalisation
    _ideal_density = n_atoms / (CELL_LENGTH * CELL_LENGTH)  # 2D density
    _normalisation = _shell_area * _ideal_density

    all_rdf = []
    for probs in P.T:
        P_adjusted = np.repeat(probs / n_atoms, length)
        counts, _ = np.histogram(
            d_stack, bins=_bins, density=False, weights=P_adjusted
        )
        rdf = counts / _normalisation
        all_rdf.append(rdf)

    return T, bin_centers, all_rdf
