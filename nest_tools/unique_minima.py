import numpy as np
import polars as pl
from ase.neighborlist import neighbor_list


def mean_filter(data, l=1):
    """Smoothens 1D-array

    :param data: raw data
    :type data: array
    :param l: array widow to smooth over (2l+1 window), defaults to 1
    :type l: int, optional
    :return: smoothend data
    :rtype: array
    """
    new_data = np.zeros_like(data)
    for i in range(len(data)):
        if i < l:
            new_data[i] = np.mean(data[: l + 1])
        elif len(data) - i <= l:
            new_data[i] = np.mean(data[i - l :])
        else:
            new_data[i] = np.mean(data[i - l : i + l + 1])
    return new_data



def find_unique_minima(atoms):
    """Finds the unique minimas using pairwise distances, only works on non-noisy data

    :param atoms: atoms
    :type atoms: ASE atoms
    """

    def get_lim(deltas, at_id):
        # Gets the limit from all the summed up distances by building a histogram
        del deltas[at_id]
        counts, bin_edges = np.histogram(
            deltas,
            bins=np.linspace(np.min(deltas), np.min(deltas) + 2, 101),
            density=True,
        )
        counts = mean_filter(counts)
        return bin_edges[np.where(counts == 0)[0][1]]

    def get_new_minima(df, at_id, i, chunk):
        # updates the array of minima
        data = np.sum(
            np.abs(
                np.array(df[chunk].to_list())
                - np.repeat([df[chunk][int(at_id)]], df.shape[0], axis=0)
            ),
            axis=1,
        )
        lim = get_lim(list(data), at_id)
        return np.where(data < lim, i, df["min_id"].to_list())

    df = pl.DataFrame({"atoms": atoms, "min_id": np.zeros(len(atoms), dtype=int)})
    # df = df.with_columns(pl.col('atoms').map_elements(lambda t: np.sort(neighbor_list('d', t[t.numbers==8], 20, self_interaction=True))[:40],return_dtype=pl.Object).alias("d_O"))
    # df = df.with_columns(pl.col('atoms').map_elements(lambda t: np.sort(neighbor_list('d', t[t.numbers==29], 20, self_interaction=True))[:100],return_dtype=pl.Object).alias("d_Cu"))
    df = df.with_columns(
        pl.col("atoms")
        .map_elements(
            lambda t: np.sort(neighbor_list("d", t, 5, self_interaction=True))[:200],
            return_dtype=pl.Object,
        )
        .alias("d_all")
    )
    unique = []
    i = 1
    while df["min_id"].min() == 0:
        at_id = np.where(df["min_id"].to_numpy() == 0)[0][0]
        unique.append(atoms[at_id])
        min_id = get_new_minima(df, at_id, i, "d_all")
        df = df.with_columns(pl.Series(name="min_id", values=min_id))
        i += 1
    return unique
