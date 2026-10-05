"""
util_funcs -- spatial tessellation helpers for step 1 of the pipeline.

`1_calculate_weekly_emissions.py` needs two helpers from the author's MSc
thesis module `util_funcs`, which was never committed to that repository. They
are reproduced here verbatim in behaviour so that step 1 is importable without
any database access.

Only the tessellation logic is included. The original module also contained a
PostgreSQL downloader (`download_trajectories`) for the restricted vehicular GPS
traces held in the KDD Lab database; that code is deliberately omitted, and with
it any credential handling. Nothing in this file reads or requires credentials.

The two functions depend only on `scikit-mobility`, which the rest of step 1
already requires.
"""

from skmob.tessellation import tilers


def download_square_tessellation(cell_size=1500, region='Greater London'):
    """
    Download a square spatial tessellation of the `region` indicated in input.
    Each cell in the spatial tesselation has side of dimension `cell_size`.

    Parameters
    ----------
    cell_size : int
        the size of the side of each cell. Default: 1500

    region : str
        the name of the region. Default: 'Greater London'

    Returns
    -------
    GeoDataFrame
        a GeoDataFrame describing the spatial tessellation
    """
    # Build a tessellation over the city
    return tilers.tiler.get("squared", base_shape=region, meters=cell_size)


def select_trajectories_within_tessellation(tdf, tessellation):
    """
    Select only the trajectories that fall entirely within the spatial tessellation.

    Parameters
    ----------
    tdf : TrajDataFrame
        the TrajDataFrame the describe the trajectories

    tessellation : GeoDataFrame
        the GeoDataFrame describing the spatial tessellation

    Returns
    -------
    TrajDataFrame
        a TrajDataFrame that contains all and only the trajectories that fall
        entirely within the spatial tessellation
    """
    # map each point to the corresponding tile in the tessellation
    tdf_mapped = tdf.mapping(tessellation, remove_na=True)

    def check_nan(tdf):
        """
        Check if there is a NaN in a trajectory's TrajDataFrame.

        Parameters
        ----------
        tdf : TrajDataFrame
            contains info about a user's trajectory

        Returns
        -------
        TrajDataFrame
            the a user's trajectory
        """
        if not tdf['tile_ID'].isna().values.any():
            return tdf
        else:
            tdf['tile_ID'] = -1
            return tdf

    if 'tid' in tdf_mapped.columns:
        grouped_tdf = tdf_mapped.groupby(['uid', 'tid']).apply(lambda uid_tid_tdf: check_nan(uid_tid_tdf))
    else:
        grouped_tdf = tdf_mapped.groupby('uid').apply(lambda uid_tdf: check_nan(uid_tdf))
    return grouped_tdf.loc[grouped_tdf['tile_ID'] != -1]
