
from ase.neighborlist import neighbor_list
from vesin import NeighborList as vesin_nl
import numpy as np
import networkx as nx
import math, pynauty, hashlib
from ase import Atoms
from typing import Optional, Callable

from tqdm import tqdm





def get_connectivity(atoms):

    def compute_graph(at):
        ids = [i.index for i in at if i.position[2]<5.1]
        for i in sorted(ids,reverse=True):
            del at[i]
        G = nx.Graph()
        G.add_nodes_from(range(len(at)))
        
        edges = np.stack(neighbor_list('ij',at, 3.0))
        unique_edges = np.unique(np.sort(edges.T,axis=1),axis=0)
        G.add_edges_from(unique_edges)
        degree = G.degree
        node_type = np.zeros(len(G))
        for i in G:
            node_type[i]=degree[i]
        return node_type
    
    data = []
    for at in atoms:
        data.append(compute_graph(at)) 

    env = np.zeros((len(data),int(np.max(np.unique(data))+1)))

    for i,val in enumerate(data):
        key, count = np.unique(val, return_counts=True)
        for j, k in enumerate(key):
            env[i,int(k)] = count[j]

    return env


def get_rdf_ase(atoms):
    data = []
    for at in atoms:
        ids = [i.index for i in at if i.position[2]<5.1]
        for i in sorted(ids,reverse=True):
            del at[i]
        data.append(neighbor_list('d',at, 5.0))
    
    return(data)



def get_atoms(partition):

    data = []
    for p in tqdm(partition):
        data.append(Atoms(numbers=p.atoms.get_atomic_numbers(),positions=p.atoms.positions, cell = p.atoms.cell))
    
    return(data)




def get_rdf(partition):
    nl_fn = vesin_nl(float(5.0), full_list=True, sorted=False)

    data = []
    for p in tqdm(partition):
        Z = p.atoms.get_atomic_numbers()
        d = nl_fn.compute(
            quantities="d",
            periodic=True,
            box=np.array(p.AtomPositionManager.latticeVectors),
            points=np.array([val for i, val in enumerate(p.AtomPositionManager.positions) if val[2]>5.1 and Z[i]==29]),
        )
        data.append(d[0])
    
    return(data)


def get_rdf_full(partition):
    nl_fn = vesin_nl(float(10.0), full_list=True, sorted=False)

    data = []
    for p in tqdm(partition):
        d = nl_fn.compute(
            quantities="d",
            periodic=True,
            box=np.array(p.atoms.cell),
            points=np.array(p.atoms.positions),
        )
        data.append(d[0])
    
    return(data)





def get_graph(partition):
    # nl_fn = vesin_nl(float(3.0), full_list=True, sorted=False) # Cu
    nl_fn = vesin_nl(float(2.3), full_list=True, sorted=False) # CuO

    def compute_graph(p):
        positions = np.array([i for i in p.AtomPositionManager.positions if i[2]>5.1])
        indices = np.array(nl_fn.compute(
            quantities="ij",
            periodic=True,
            box=np.array(p.AtomPositionManager.latticeVectors),
            points=positions,
        ))
        edges = np.unique(np.sort(indices.T,axis=1),axis=0)

        G = nx.Graph()
        G.add_nodes_from(range(len(positions)))
        G.add_edges_from(edges)
        return G

        
    data = []
    for p in tqdm(partition):
        data.append(compute_graph(p))

    return data


def get_graph_CuO(partition):
    # nl_fn = vesin_nl(float(3.0), full_list=True, sorted=False) # Cu
    nl_fn = vesin_nl(float(2.5), full_list=False, sorted=False) # CuO
    

    def compute_graph(p):
        idx = np.array([i for i, val in enumerate(p.atoms.positions) if val[2]>5.1],dtype=int)
        positions = np.array(p.atoms.positions[idx])
        Z = np.array(p.atoms.get_atomic_numbers()[idx])
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
    for p in tqdm(partition):
        data.append(compute_graph(p))

    return data



def get_graph_count(partition):
    nl_fn = vesin_nl(float(2.3), full_list=True, sorted=False) # 2.3 for CuO

    def compute_graph(p):
        positions = np.array([i for i in p.AtomPositionManager.positions if i[2]>5.1])
        indices = np.array(nl_fn.compute(
            quantities="ij",
            periodic=True,
            box=np.array(p.AtomPositionManager.latticeVectors),
            points=positions,
        ))
        edges = np.unique(np.sort(indices.T,axis=1),axis=0)

        G = nx.Graph()
        G.add_nodes_from(range(len(positions)))
        G.add_edges_from(edges)
        return G

        
    data = []
    for p in tqdm(partition):
        data.append((nx.number_connected_components(compute_graph(p))))

    return data




def _build_integer_modes_sphere(kmax: int) -> np.ndarray:
    """
    Build an integer reciprocal grid M ⊂ Z^3 inside a sphere of radius kmax,
    excluding the origin. The list is deterministic (lexicographic order).

    Parameters
    ----------
    kmax : int
        Radius in integer space. kmax=3 → typically ~ 100 modes.

    Returns
    -------
    np.ndarray
        Array of shape (M, 3) with integer triples (h, k, l).
    """
    rng = range(-kmax, kmax + 1)
    modes = []
    for h in rng:
        for k in rng:
            if h == 0 and k == 0:
                continue
            if h*h + k*k <= kmax*kmax:
                    modes.append((h, k, 0))
    M = np.asarray(modes, dtype=np.int32)
    M = M[np.lexsort((M[:, 2], M[:, 1], M[:, 0]))]
    return M

def _is_nonperiodic(apm) -> bool:
    # Prefer explicit flag if your APM exposes it
    pbc = getattr(apm, "pbc", None)           # e.g. (False, False, False)
    if isinstance(pbc, (tuple, list)) and not any(pbc):
        return True
    # Fallback: missing/degenerate lattice → treat as non-periodic
    try:
        lat = np.asarray(apm.latticeVectors, float)
        return not np.isfinite(lat).all() or abs(np.linalg.det(lat)) < 1e-9
    except Exception:
        return True


def _structure_factor_factory(
    *,
    # Fourier sampling
    kmax: int = 3,
    modes: Optional[np.ndarray] = None,   # custom (M,3) integer modes; overrides kmax when given
    per_species: bool = True,             # accumulate |S| per atomic number channel
    # Performance / numerical robustness
    chunk_size: int = 50000,              # number of atoms per chunk in accumulation
    **kwargs,
) -> Callable[[object], str]:
    """
    Factory for a fast, translation-invariant global hash based on the modulus
    of structure factors on a small set of integer reciprocal modes.

    Mathematical idea
    ------------------
    For a periodic structure with fractional coordinates F ∈ [0,1)^{N×3},
    the (unnormalized) structure factor on an integer mode m ∈ Z^3 is:
        S(m) = Σ_i exp(i 2π m·F_i).
    A rigid translation by t (fractional) sends F_i → F_i + t, hence
        S'(m) = exp(i 2π m·t) S(m),
    so the modulus |S(m)| is *exactly translation invariant*. We compute |S|
    on a small, fixed set of modes (a sphere in Z^3 of radius kmax), optionally
    per species, lightly quantize, and hash the integer tensor.

    Complexity and scaling
    ----------------------
    The computation is O(N * M) where M is the number of modes (dozens to low
    hundreds). This scales linearly in N and is robust for N >> 10^3. We process
    atoms in chunks to bound memory usage.

    Assumptions
    -----------
    The container exposes `AtomPositionManager` (APM) with:
      - `atomPositions` (Cartesian, shape (N,3), floats, current),
      - `latticeVectors` (3×3, your codebase stores as *columns* a,b,c),
      - `get_atomic_numbers()` → (N,) array-like of ints,
      - optional `.E` (total energy).
    We do not rely on cached fractional coordinates; we recompute them from the
    *current* Cartesian positions and the lattice each call.

    Parameters
    ----------
    kmax : int
        Sphere radius in integer reciprocal space (controls M).
    modes : np.ndarray, optional
        Custom integer modes (M,3). If given, kmax is ignored.
    per_species : bool
        If True, produce a channel per atomic number (more discriminative).
        If False, aggregate all atoms together (faster/shorter descriptor).
    chunk_size : int
        Chunk length for the per-mode accumulation loop. Increase for speed
        if RAM allows; decrease to reduce memory footprint.

    Returns
    -------
    Callable[[object], str]
        A function `hash_fn(container) -> hex_string` producing a SHA-256 digest.
    """
    TWO_PI = 2.0 * math.pi
    EPS = 1e-12  # small offset to avoid hitting exactly 0/1 after modulo

    # Build / validate reciprocal modes
    if modes is None:
        M = _build_integer_modes_sphere(kmax)
    else:
        M = np.asarray(modes, dtype=np.float64)
        #if M.ndim != 2 or M.shape[1] != 3 or M.shape[0] == 0:
        #    raise ValueError("modes must have shape (M,3) with M>0")
        # Make deterministic: sort lexicographically
        #M = M[np.lexsort((M[:, 2], M[:, 1], M[:, 0]))]
    M_T = M.T  # (3, M)
    M_count = int(M.shape[0])

    
    def sf_calc(container) -> str:
        apm = container.AtomPositionManager

        # 1) Detect periodicity robustly.
        if _is_nonperiodic(apm):
             is_nonpbc = True
        else:
             is_nonpbc = False

        # 2) Basic properties
        
        Z = np.asarray([apm.get_atomic_numbers()[i] for i, val in enumerate(apm.atomPositions) if val[2]>5.1], np.int32)
        N = int(Z.shape[0])
        if N == 0:
            # Empty container → hash a fixed token
            return None
        
        # 4) Non-PBC fallback: reuse a fast, centered integer multiset (no neighbors).
        if is_nonpbc:
            print('Error')
            return None
        
        # 5) PBC path: recompute fractionals from current Cartesian + lattice.
        A_cols = np.asarray(apm.latticeVectors, float)   # your APM stores columns a,b,c
        A = A_cols.T                                     # rows (a,b,c) for math
        X = np.asarray([i for i in apm.atomPositions if i[2]>5.1], float)
        #X = np.asarray(apm.atomPositions, float)         # current Cartesian
        FinvT = np.linalg.inv(A).T
        F = (X @ FinvT + EPS) % 1.0                      # fractional coords ∈ [0,1)

        
        # 6) Prepare species channels (optional but recommended).
        if per_species:
            species, inv = np.unique(Z, return_inverse=True)  # species sorted ascending
            C = int(species.shape[0])
        else:
            species = np.array([0], dtype=np.int32)
            inv = np.zeros(N, dtype=np.int32)
            C = 1

        # 7) Accumulate structure factors S per species and per mode.
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

        # 8) Normalize per species to keep the descriptor scale-stable with N.
        Nz = np.bincount(inv, minlength=C).astype(np.float64)  # #atoms per species
        denom = np.sqrt(np.maximum(Nz, 1.0))[:, None]          # (C, 1)
        P = np.hypot(S_re, S_im) /denom                     # (C, M), translation-invariant
        
        
        return np.array([S_re, S_im, P])

    return sf_calc


def get_sf(partition):
    sf = _structure_factor_factory(modes = (np.array([1,0,0])[:,None]*np.linspace(1,5,5)).T)
    sf_2 = _structure_factor_factory(modes = (np.array([0,1,0])[:,None]*np.linspace(1,5,5)).T)

    data = []
    for p in tqdm(partition):
        data.append(np.mean([sf(p), sf_2(p)],axis=0))

    return np.array(data)



def canonical_hash(G: nx.Graph) -> str:
    """
    Exact isomorphism-invariant hash via nauty canonical labeling.
    Zero false positives — two graphs get the same hash IFF they are isomorphic.
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

    

