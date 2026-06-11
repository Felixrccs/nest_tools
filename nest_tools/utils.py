import hashlib
import math
from typing import Callable

import networkx as nx
import numpy as np
import pynauty
from ase import Atoms
from sage_lib.partition.Partition import Partition
from tqdm import tqdm
from vesin import NeighborList as vesin_nl

# Atoms with a z-coordinate above this value (Å) are treated as the
# surface layer in the analyses below.
Z_CUTOFF = 5.1
# Atomic number of copper.
CU = 29




def _iter_containers(partition, batch_size: int = 2000):
    """
    Iterate over the containers of a partition with batched backend reads.

    On on-disk (hybrid) partitions, structures are materialized in batches of
    ``batch_size`` to avoid a separate backend read per entry. Plain sequences
    of containers (e.g. a sliced partition, which yields a list) are iterated
    directly.

    Parameters
    ----------
    partition : Partition or sequence
        Collection of structure entries.
    batch_size : int
        Number of structures read from the backend per batch.

    Yields
    ------
    SingleRun
        The containers, in partition order.
    """
    if not hasattr(partition, "materialize_by_ids"):
        yield from tqdm(partition)
        return

    all_ids = partition.get_ids()
    for start in tqdm(range(0, len(all_ids), batch_size)):
        batch_ids = all_ids[start:start + batch_size]
        yield from partition.materialize_by_ids(batch_ids, dedup=False, sort_backend_reads=True)


def get_atoms(partition: Partition, batch_size: int = 2000) -> list[Atoms]:
    """
    Extract ASE Atoms objects from each entry in a partition.

    Structures are materialized from the backend in batches to avoid a
    separate read per entry on on-disk (hybrid) partitions.

    Parameters
    ----------
    partition : Partition
        Collection of structure entries.
    batch_size : int
        Number of structures read from the backend per batch.

    Returns
    -------
    list[Atoms]
        ASE Atoms objects with atomic numbers, positions, and cell.
    """
    data = []
    for p in _iter_containers(partition, batch_size):
        data.append(Atoms(numbers=p.atoms.get_atomic_numbers(), positions=p.atoms.positions, cell=p.atoms.cell))

    return data


def get_z(partition: Partition, batch_size: int = 2000) -> np.ndarray:
    """
    Compute the mean and maximum z-position of atoms above z=5.1 Å per frame.

    For each frame, selects atoms with a z-coordinate greater than 5.1 Å and
    records the mean and maximum of those z-coordinates. Structures are
    materialized from the backend in batches to avoid a separate read per
    entry on on-disk (hybrid) partitions.

    Parameters
    ----------
    partition : Partition
        Collection of structure entries.
    batch_size : int
        Number of structures read from the backend per batch.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(partition), 2)`` with columns
        ``(mean_z, max_z)`` per frame.
    """
    data = []
    for p in _iter_containers(partition, batch_size):
        pos = p.AtomPositionManager.positions[:, 2]
        sub_pos = pos[pos > Z_CUTOFF]
        data.append([np.mean(sub_pos), np.max(sub_pos)])

    return np.array(data)


def get_dist(partition: Partition, batch_size: int = 2000) -> list[np.ndarray]:
    """
    Compute pairwise distances for Cu atoms (Z=29) above z=5.1 Å.

    Uses a neighbor list with a 5.0 Å cutoff under periodic boundary conditions.
    Structures are materialized from the backend in batches to avoid a separate
    read per entry on on-disk (hybrid) partitions.

    Parameters
    ----------
    partition : Partition
        Collection of structure entries.
    batch_size : int
        Number of structures read from the backend per batch.

    Returns
    -------
    list[np.ndarray]
        Pairwise distances for each entry in the partition.
    """
    nl_fn = vesin_nl(float(5.0), full_list=True, sorted=False)

    data = []
    for p in _iter_containers(partition, batch_size):
        Z = np.asarray(p.atoms.get_atomic_numbers())
        pos = np.asarray(p.AtomPositionManager.positions)
        mask = (pos[:, 2] > Z_CUTOFF) & (Z == CU)
        d = nl_fn.compute(
            quantities="d",
            periodic=True,
            box=np.array(p.AtomPositionManager.latticeVectors),
            points=pos[mask],
        )
        data.append(d[0])

    return data






def get_graph_CuO(partition: Partition, batch_size: int = 2000) -> list[nx.Graph]:
    """
    Build heteroatomic neighbor graphs for atoms above z=5.1 Å (CuO surface layer).

    Uses a 2.5 Å cutoff neighbor list. Only edges between atoms of different
    species (e.g. Cu-O bonds) are retained. Structures are materialized from the
    backend in batches to avoid a separate read per entry on on-disk (hybrid)
    partitions.

    Parameters
    ----------
    partition : Partition
        Collection of structure entries.
    batch_size : int
        Number of structures read from the backend per batch.

    Returns
    -------
    list[nx.Graph]
        Undirected graphs with only heteroatomic edges, one per entry.
    """
    nl_fn = vesin_nl(float(2.5), full_list=False, sorted=False) # CuO


    def compute_graph(p):
        all_positions = np.asarray(p.atoms.positions)
        idx = np.where(all_positions[:, 2] > Z_CUTOFF)[0]
        positions = all_positions[idx]
        Z = np.asarray(p.atoms.get_atomic_numbers())[idx]
        indices = np.array(nl_fn.compute(
            quantities="ij",
            periodic=True,
            box=np.array(p.AtomPositionManager.latticeVectors),
            points=positions,
        ))
        edges = indices.T[Z[indices[0]] != Z[indices[1]]]

        G = nx.Graph()
        G.add_nodes_from(range(len(positions)))
        G.add_edges_from(edges)
        return G


    data = []
    for p in _iter_containers(partition, batch_size):
        data.append(compute_graph(p))

    return data






def canonical_hash(G: nx.Graph) -> str:
    """
    Exact isomorphism-invariant hash via nauty canonical labeling.

    Two graphs receive the same hash if and only if they are isomorphic
    (up to SHA-256 collisions). Node labels are irrelevant; only the
    topology matters.

    Parameters
    ----------
    G : nx.Graph
        Graph to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest of the nauty canonical certificate.
    """
    nodes = sorted(G.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}
    n     = len(nodes)

    adjacency = {
        idx[u]: {idx[v] for v in G.neighbors(u)}
        for u in G.nodes()
    }

    g = pynauty.Graph(
        number_of_vertices=n,
        directed=G.is_directed(),
        adjacency_dict=adjacency,
    )
    cert = pynauty.certificate(g)   # canonical certificate bytes
    return hashlib.sha256(cert).hexdigest()



def _structure_factor(
    *,
    # Fourier sampling
    modes: np.ndarray,                    # (M,3) integer reciprocal modes
    per_species: bool = True,             # accumulate |S| per atomic number channel
    # Performance / numerical robustness
    chunk_size: int = 50000,              # number of atoms per chunk in accumulation
) -> Callable[[object], np.ndarray]:
    """
    Factory for a fast, translation-invariant structure-factor descriptor
    evaluated on a small set of integer reciprocal modes. The implementation
    is adapted from https://doi.org/10.26434/chemrxiv.15000180/v1. 

    Mathematical idea
    -----------------
    For a periodic structure with fractional coordinates F ∈ [0,1)^{N×3},
    the (unnormalized) structure factor on an integer mode m ∈ Z^3 is:
        S(m) = Σ_i exp(i 2π m·F_i).
    A rigid translation by t (fractional) sends F_i → F_i + t, hence
        S'(m) = exp(i 2π m·t) S(m),
    so the modulus |S(m)| is *exactly translation invariant*. We compute S
    on the small, fixed set of supplied modes, optionally per species.

    Complexity and scaling
    ----------------------
    The computation is O(N * M) where M is the number of modes (dozens to low
    hundreds). This scales linearly in N and is robust for N >> 10^3. We process
    atoms in chunks to bound memory usage.

    Assumptions
    -----------
    The container exposes `AtomPositionManager` (APM) with:
      - `atomPositions` (Cartesian, shape (N,3), floats, current),
      - `latticeVectors` (3×3, stored as *columns* a, b, c),
      - `get_atomic_numbers()` → (N,) array-like of ints.
    Only atoms above z=Z_CUTOFF enter the descriptor. We do not rely on cached
    fractional coordinates; we recompute them from the *current* Cartesian
    positions and the lattice each call.

    Parameters
    ----------
    modes : np.ndarray
        Integer reciprocal modes of shape (M, 3).
    per_species : bool
        If True, produce a channel per atomic number (more discriminative).
        If False, aggregate all atoms together (faster/shorter descriptor).
    chunk_size : int
        Chunk length for the per-mode accumulation loop. Increase for speed
        if RAM allows; decrease to reduce memory footprint.

    Returns
    -------
    Callable[[object], np.ndarray]
        A function `sf_calc(container)` returning an array of shape
        ``(3, C, M)`` stacking the real part, imaginary part, and the
        per-species-normalized modulus of the structure factors, where C is
        the number of species channels.
    """
    TWO_PI = 2.0 * math.pi
    EPS = 1e-12  # small offset to avoid hitting exactly 0/1 after modulo

    # Reciprocal modes
    M = np.asarray(modes, dtype=np.float64)
    M_T = M.T  # (3, M)
    M_count = int(M.shape[0])

    def compute(container) -> np.ndarray:
        apm = container.AtomPositionManager

        # 1) Select surface atoms (z > Z_CUTOFF) and their species.
        positions = np.asarray(apm.atomPositions, float)
        mask = positions[:, 2] > Z_CUTOFF
        Z = np.asarray(apm.get_atomic_numbers(), np.int32)[mask]
        N = int(Z.shape[0])

        # 2) Recompute fractionals from current Cartesian + lattice.
        A_cols = np.asarray(apm.latticeVectors, float)   # APM stores columns a, b, c
        A = A_cols.T                                     # rows (a, b, c) for math
        X = positions[mask]
        FinvT = np.linalg.inv(A).T
        F = (X @ FinvT + EPS) % 1.0                      # fractional coords ∈ [0,1)

        # 3) Prepare species channels (optional but recommended).
        if per_species:
            species, inv = np.unique(Z, return_inverse=True)  # species sorted ascending
            C = int(species.shape[0])
        else:
            species = np.array([0], dtype=np.int32)
            inv = np.zeros(N, dtype=np.int32)
            C = 1

        # 4) Accumulate structure factors S per species and per mode.
        #    S_re, S_im have shape (C, M); we sum in float64 for stability.
        S_re = np.zeros((C, M_count), dtype=np.float64)
        S_im = np.zeros((C, M_count), dtype=np.float64)

        # Chunked accumulation to keep memory bounded: (chunk×3) @ (3×M) = (chunk×M)
        for s in range(0, N, max(1, int(chunk_size))):
            e = min(N, s + chunk_size)
            Fi = F[s:e]               # (chunk, 3)
            inv_i = inv[s:e]          # (chunk,)
            phase = TWO_PI * (Fi @ M_T)     # (chunk, M)
            Ei_re = np.cos(phase)           # (chunk, M)
            Ei_im = np.sin(phase)           # (chunk, M)

            # Accumulate by species using bincount per mode (fast and vectorized).
            # Loop over modes only (typically ~100), not over atoms.
            for m in range(M_count):
                S_re[:, m] += np.bincount(inv_i, weights=Ei_re[:, m], minlength=C)
                S_im[:, m] += np.bincount(inv_i, weights=Ei_im[:, m], minlength=C)

        # 5) Normalize per species to keep the descriptor scale-stable with N.
        Nz = np.bincount(inv, minlength=C).astype(np.float64)  # #atoms per species
        denom = np.sqrt(np.maximum(Nz, 1.0))[:, None]          # (C, 1)
        P = np.hypot(S_re, S_im) / denom                       # (C, M), translation-invariant

        return np.array([S_re, S_im, P])

    return compute


def get_sf(partition: Partition, batch_size: int = 2000) -> np.ndarray:
    """
    Compute structure factor descriptors for each entry in a partition.

    Evaluates structure factors along the [100] and [010] directions for
    integer mode indices 1–5, then averages the two contributions. Structures
    are materialized from the backend in batches to avoid a separate read per
    entry on on-disk (hybrid) partitions.

    Parameters
    ----------
    partition : Partition
        Collection of structure entries.
    batch_size : int
        Number of structures read from the backend per batch.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(partition), 3, C, M)`` containing the averaged
        real part, imaginary part, and modulus of the structure factors,
        where C is the number of species channels and M the number of modes.
    """
    sf = _structure_factor(modes = (np.array([1,0,0])[:,None]*np.linspace(1,5,5)).T)
    sf_2 = _structure_factor(modes = (np.array([0,1,0])[:,None]*np.linspace(1,5,5)).T)

    data = []
    for p in _iter_containers(partition, batch_size):
        data.append(np.mean([sf(p), sf_2(p)], axis=0))

    return np.array(data)
