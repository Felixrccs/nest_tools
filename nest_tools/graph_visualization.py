import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase import Atoms
from ase.visualize.plot import Matplotlib, plot_atoms as ase_plot_atoms
from .utils import get_graph_CuO

from sage_lib.partition.Partition import Partition



def get_graph_edges(
    p: Partition,
    z_cutoff: float = 5.1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return edge segments for a graph in the original Cartesian xyz-coordinates.

    Periodic edges are split into two half-segments using the minimum-image
    convention in the xy-plane. The z coordinate is kept as-is.

    Parameters
    ----------
    p : Partition
        a sage_lib Partition
    z_cutoff : float
        z-coordinate threshold used when the graph was built.

    Returns
    -------
    segments : list of (start, end) pairs
        Each element is a tuple of two 1-D arrays of shape (3,) in Cartesian
        Angstrom (x, y, z).  Wrapping edges contribute two segments; non-wrapping
        edges contribute one.
    """
    atm  = p[0].atoms
    mask = atm.positions[:, 2] > z_cutoff
    pos3d = atm.positions[mask]

    cell = np.asarray(atm.cell)
    cell2d = cell[:2, :2]
    periodic = abs(np.linalg.det(cell2d)) > 1e-9

    G = get_graph_CuO(p[0:1])[0]
    segments = []
    for u, v in G.edges():
        p0 = pos3d[u]
        p1 = pos3d[v]
        diff = p1 - p0

        if periodic:
            frac = np.linalg.solve(cell2d.T, diff[:2])
            frac -= np.round(frac)
            diff_mi_xy = cell2d.T @ frac
            diff_mi = np.array([diff_mi_xy[0], diff_mi_xy[1], diff[2]])
        else:
            diff_mi = diff

        wraps = np.linalg.norm(diff[:2] - diff_mi[:2]) > 1e-3

        if wraps:
            segments.append((p0.copy(), (p0 + 2 * diff_mi).copy()))
            segments.append(((p1 - 2 * diff_mi).copy(), p1.copy()))
        else:
            segments.append((p0.copy(), (p0 + diff_mi).copy()))

    return np.array(segments)


def plot_atoms_with_graph(
    atoms: Atoms,
    G: nx.Graph,
    ax=None,
    rotation: str = '',
    z_cutoff: float = 5.1,
    edge_color: str = 'white',
    edge_lw: float = 3.0,
    draw_nodes: bool = False,
    node_color: str = 'steelblue',
    node_size: float = 50.0,
    draw_cell: bool = False,
    cell_color: str = 'black',
    cell_lw: float = 1.5,
    **plot_atoms_kwargs,
):
    """
    Plot an ASE Atoms object with a NetworkX graph overlaid in matplotlib.

    Graph nodes 0..N-1 are assumed to correspond (in order) to atoms with
    z > z_cutoff, matching the convention in get_graph / get_graph_CuO.
    Periodic edges are drawn using the minimum-image displacement so they
    point toward the correct periodic image rather than crossing the cell.

    Parameters
    ----------
    atoms : Atoms
        Full ASE Atoms object (all atoms in the cell).
    G : nx.Graph
        NetworkX graph whose nodes map to atoms[z > z_cutoff], in order.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure/axes is created if None.
    rotation : str
        Rotation string passed to plot_atoms (e.g. '45x,30y').
        Default '' gives a top-down view (xy-plane).
    z_cutoff : float
        z-coordinate threshold used when the graph was built.
    edge_color : str
        Color for graph edges.
    edge_lw : float
        Line width for graph edges.
    draw_nodes : bool
        If True, draw scatter markers on graph node positions (on top of
        the atom circles already drawn by plot_atoms).
    node_color : str
        Marker color for graph nodes (only used when draw_nodes=True).
    node_size : float
        Marker size for graph nodes (only used when draw_nodes=True).
    draw_cell : bool
        If True, draw the unit cell boundary using matplotlib instead of ASE.
    cell_color : str
        Color of the cell boundary lines.
    cell_lw : float
        Line width of the cell boundary.
    **plot_atoms_kwargs
        Additional keyword arguments forwarded to ase plot_atoms
        (e.g. radii, colors, scale).

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    # Set default line width to 2 points
    plt.rcParams['lines.linewidth'] = 0.5

    # Suppress ASE's own cell drawing; we handle it below with matplotlib.
    atoms_kwargs = {**plot_atoms_kwargs, 'show_unit_cell': 0}

    # Build the same internal transform that plot_atoms uses, so that graph
    # coordinates end up in the same display space as the drawn atoms.
    # Matplotlib.__init__ does NOT draw anything (drawing happens in .write());
    # it only computes the rotation matrix, scale, and offset.
    _pv = Matplotlib(atoms, ax, rotation=rotation, **atoms_kwargs)

    # Draw atoms with ASE (creates an identical Matplotlib instance internally)
    print(atoms_kwargs)
    ase_plot_atoms(atoms, ax=ax, rotation=rotation, **atoms_kwargs)


    plt.rcParams['lines.linewidth'] = 1

    # Positions of graph nodes: atoms above z_cutoff, same ordering as get_graph
    mask = atoms.positions[:, 2] > z_cutoff
    node_pos3d = atoms.positions[mask]           # (N_nodes, 3), Cartesian Angstrom

    # Convert to display coordinates using the same transform as plot_atoms:
    #   im_pos = (pos @ rotation) * scale - offset
    node_im = _pv.to_image_plane_positions(node_pos3d)
    pos2d = node_im[:, :2]                       # (N_nodes, 2), display units

    # Cell vectors as displacements in display space (offset cancels for differences):
    #   cell_disp = (cell_row @ rotation) * scale
    cell = np.asarray(atoms.cell)
    cell_im = (cell @ _pv.rotation) * _pv.scale  # (3, 3)
    cell2d = cell_im[:2, :2]                     # in-plane a, b vectors
    periodic = abs(np.linalg.det(cell2d)) > 1e-9

    if draw_cell and periodic:
        # Cell corners in 3D: origin, a, a+b, b
        origin_3d = np.zeros((1, 3))
        corners_3d = np.vstack([origin_3d,
                                cell[[0]],
                                cell[[0]] + cell[[1]],
                                cell[[1]]])
        corners_2d = _pv.to_image_plane_positions(corners_3d)[:, :2]
        cell_patch = mpatches.Polygon(corners_2d, closed=True,
                                      fill=False, edgecolor=cell_color,
                                      linewidth=cell_lw, zorder=6 ,linestyle='dashed')
        ax.add_patch(cell_patch)

    for u, v in G.edges():
        p0 = pos2d[u]
        p1 = pos2d[v]
        diff = p1 - p0

        if periodic:
            # Minimum-image displacement in 2D fractional coordinates
            frac = np.linalg.solve(cell2d.T, diff)
            frac -= np.round(frac)
            diff_mi = cell2d.T @ frac
        else:
            diff_mi = diff

        wraps = np.linalg.norm(diff - diff_mi) > 1e-3

        if wraps:
            # Two half-segments: one exits the cell from u's side,
            # the other enters on v's side. matplotlib clips both at the axes boundary.
            end0 = p0 + diff_mi
            ax.plot([p0[0], end0[0]], [p0[1], end0[1]],
                    color=edge_color, lw=edge_lw, zorder=4, solid_capstyle='round')
            start1 = p1 - diff_mi
            ax.plot([start1[0], p1[0]], [start1[1], p1[1]],
                    color=edge_color, lw=edge_lw, zorder=4, solid_capstyle='round')
        else:
            end = p0 + diff_mi
            ax.plot([p0[0], end[0]], [p0[1], end[1]],
                    color=edge_color, lw=edge_lw, zorder=4, solid_capstyle='round')

    if draw_nodes:
        ax.scatter(pos2d[:, 0], pos2d[:, 1],
                   s=node_size, c=node_color, zorder=5)

    return ax



def plot_from_partition(ax,p):
    atm  = p[0].atoms
    slab = Atoms(numbers=atm.get_atomic_numbers(),positions = atm.positions, cell = atm.cell, pbc=True)

    symbols = ['Pd' if at.position[2] > 5.1 and at.symbol == 'Cu' else at.symbol for at in slab]

    slab.set_chemical_symbols(symbols)
    slab.positions+= np.matmul(np.array([1.,1.,0.])/16,slab.cell)
    slab.wrap()

    G = get_graph_CuO(p[0:1])[0]
    plot_atoms_with_graph(slab, G, ax=ax, z_cutoff=5.1, edge_lw=1.5, cell_lw=1.5)




