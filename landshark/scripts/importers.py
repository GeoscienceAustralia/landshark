"""Landshark importing commands."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os.path
from multiprocessing import cpu_count
from typing import List, NamedTuple, Tuple

import click
import numpy as np
import tables

from landshark import __version__, errors
from landshark import metadata as meta
from landshark.category import get_maps
from landshark.featurewrite import (
    write_categorical,
    write_continuous,
    write_coordinates,
    write_feature_metadata,
    write_target_metadata,
)
from landshark.fileio import tifnames
from landshark.normalise import get_stats
from landshark.scripts.logger import configure_logging
from landshark.shpread import (
    CategoricalShpArraySource,
    ContinuousShpArraySource,
    CoordinateShpArraySource,
)
from landshark.tifread import (
    CategoricalStackSource,
    ContinuousStackSource,
    shared_image_spec,
)
from landshark.util import mb_to_points, mb_to_rows

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    nworkers: int
    batchMB: float


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Level of logging",
)
@click.option(
    "--nworkers",
    type=click.IntRange(0, None),
    default=cpu_count(),
    help="Number of additional worker processes",
)
@click.option(
    "--batch-mb",
    type=float,
    default=10,
    help="Approximate size in megabytes of data read per " "worker per iteration",
)
@click.pass_context
def cli(ctx: click.Context, verbosity: str, nworkers: int, batch_mb: float) -> int:
    """Import features and targets into landshark-compatible formats."""
    log.info("Using a maximum of {} worker processes".format(nworkers))
    ctx.obj = CliArgs(nworkers, batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option(
    "--categorical",
    type=click.Path(exists=True),
    multiple=True,
    help="Directory containing categorical geotifs",
)
@click.option(
    "--continuous",
    type=click.Path(exists=True),
    multiple=True,
    help="Directory containing continuous geotifs",
)
@click.option(
    "--categorical_list",
    type=click.Path(exists=True),
    multiple=True,
    help="List/txt file containing categorical geotifs",
)
@click.option(
    "--continuous_list",
    type=click.Path(exists=True),
    multiple=True,
    help="List/txt file containing continuous geotifs",
)
@click.option(
    "--normalise/--no-normalise",
    is_flag=True,
    default=True,
    help="Normalise the continuous tif bands",
)
@click.option("--name", type=str, required=True, help="Name of output file")
@click.option(
    "--ignore-crs/--no-ignore-crs",
    is_flag=True,
    default=False,
    help="Ignore CRS (projection and datum) information",
)
@click.pass_context
def tifs(
    ctx: click.Context,
    categorical: Tuple[str, ...],
    continuous: Tuple[str, ...],
    categorical_list: Tuple[str, ...],
    continuous_list: Tuple[str, ...],
    normalise: bool,
    name: str,
    ignore_crs: bool,
) -> None:
    """Build a tif stack from a set of input files."""
    nworkers = ctx.obj.nworkers
    batchMB = ctx.obj.batchMB
    cat_list = list(categorical)
    con_list = list(continuous)
    cat_lists_list = list(categorical_list)
    con_lists_list = list(continuous_list)
    catching_f = errors.catch_and_exit(tifs_entrypoint)
    catching_f(nworkers, batchMB, cat_list, con_list, cat_lists_list, con_lists_list, normalise, name, ignore_crs)


def tifs_entrypoint(
    nworkers: int,
    batchMB: float,
    categorical: List[str],
    continuous: List[str],
    categorical_lists: List[str],
    continuous_lists: List[str],
    normalise: bool,
    name: str,
    ignore_crs: bool,
) -> None:
    """Entrypoint for tifs without click cruft."""
    out_filename = os.path.join(os.getcwd(), "features_{}.hdf5".format(name))

    con_filenames = tifnames(continuous, continuous_lists)
    cat_filenames = tifnames(categorical, categorical_lists)
    log.info("Found {} continuous TIF files".format(len(con_filenames)))
    log.info("Found {} categorical TIF files".format(len(cat_filenames)))
    has_con = len(con_filenames) > 0
    has_cat = len(cat_filenames) > 0
    all_filenames = con_filenames + cat_filenames
    if not len(all_filenames) > 0:
        raise errors.NoTifFilesFound()

    N_con, N_cat = None, None
    con_meta, cat_meta = None, None
    spec = shared_image_spec(all_filenames, ignore_crs)

    with tables.open_file(out_filename, mode="w", title=name) as outfile:
        if has_con:
            con_source = ContinuousStackSource(spec, con_filenames)
            ndims_con = con_source.shape[-1]
            con_rows_per_batch = mb_to_rows(batchMB, spec.width, ndims_con, 0)
            N_con = con_source.shape[0] * con_source.shape[1]
            N = N_con
            log.info("Continuous missing value set to {}".format(con_source.missing))
            stats = None
            if normalise:
                stats = get_stats(con_source, con_rows_per_batch)
                sd = stats[1]
                if any(sd == 0.0):
                    raise errors.ZeroDeviation(sd, con_source.columns)
                log.info("Writing normalised continuous data to output file")
            else:
                log.info("Writing unnormalised continuous data to output file")
            con_meta = meta.ContinuousFeatureSet(
                labels=con_source.columns, missing=con_source.missing, stats=stats
            )
            write_continuous(con_source, outfile, nworkers, con_rows_per_batch, stats)

        if has_cat:
            cat_source = CategoricalStackSource(spec, cat_filenames)
            N_cat = cat_source.shape[0] * cat_source.shape[1]
            N = N_cat
            if N_con and N_cat != N_con:
                raise errors.ConCatNMismatch(N_con, N_cat)

            ndims_cat = cat_source.shape[-1]
            cat_rows_per_batch = mb_to_rows(batchMB, spec.width, 0, ndims_cat)
            log.info("Categorical missing value set to {}".format(cat_source.missing))
            catdata = get_maps(cat_source, cat_rows_per_batch)
            maps, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in maps])
            log.info("Writing mapped categorical data to output file")
            cat_meta = meta.CategoricalFeatureSet(
                labels=cat_source.columns,
                missing=cat_source.missing,
                nvalues=ncats,
                mappings=maps,
                counts=counts,
            )
            write_categorical(cat_source, outfile, nworkers, cat_rows_per_batch, maps)
        m = meta.FeatureSet(
            continuous=con_meta, categorical=cat_meta, image=spec, N=N, halfwidth=0
        )
        write_feature_metadata(m, outfile)
    log.info("Tif import complete")


@cli.command()
@click.option(
    "--record",
    type=str,
    multiple=True,
    required=True,
    help="Label of record to extract as a target",
)
@click.option(
    "--group_col",
    type=str,
    multiple=False,
    required=False,
    help="Group value to extract for each target",
)
@click.option(
    "--shapefile",
    type=click.Path(exists=True),
    required=True,
    help="Path to .shp file for reading",
)
@click.option("--name", type=str, required=True, help="Name of output file")
@click.option(
    "--every",
    type=int,
    default=1,
    help="Subsample (randomly)" " by this factor, e.g. every 2 samples half the points",
)
@click.option(
    "--dtype",
    type=click.Choice(["continuous", "categorical"]),
    required=True,
    help="The type of the targets",
)
@click.option(
    "--normalise",
    is_flag=True,
    help="Normalise each target." " Only relevant for continuous targets.",
)
@click.option(
    "--random_seed",
    type=int,
    default=666,
    help="The random seed " "for shuffling targets on import",
)
@click.pass_context
def targets(
    ctx: click.Context,
    shapefile: str,
    record: Tuple[str, ...],
    group_col: str,
    name: str,
    every: int,
    dtype: str,
    normalise: bool,
    random_seed: int,
) -> None:
    """Build target file from shapefile."""
    record_list = list(record)
    categorical = dtype == "categorical"
    batchMB = ctx.obj.batchMB
    catching_f = errors.catch_and_exit(targets_entrypoint)
    catching_f(
        batchMB,
        shapefile,
        record_list,
        group_col,
        name,
        every,
        categorical,
        normalise,
        random_seed,
    )


def targets_entrypoint(
    batchMB: float,
    shapefile: str,
    records: List[str],
    group_col: str,
    name: str,
    every: int,
    categorical: bool,
    normalise: bool,
    random_seed: int,
) -> None:
    """Targets entrypoint without click cruft."""
    log.info("Loading shapefile targets")
    out_filename = os.path.join(os.getcwd(), "targets_{}.hdf5".format(name))
    nworkers = 0  # shapefile reading breaks with concurrency

    with tables.open_file(out_filename, mode="w", title=name) as h5file:
        log.info("Reading shapefile point coordinates")
        cocon_src = CoordinateShpArraySource(shapefile, random_seed)
        cocon_batchsize = mb_to_points(batchMB, ndim_con=0, ndim_cat=0, ndim_coord=2)
        write_coordinates(cocon_src, h5file, cocon_batchsize)
        if group_col:
            write_group_data(batchMB, group_col, nworkers, h5file, random_seed, shapefile)

        if categorical:
            log.info("Reading shapefile categorical records")
            cat_source = CategoricalShpArraySource(shapefile, records, random_seed)
            cat_batchsize = mb_to_points(
                batchMB, ndim_con=0, ndim_cat=cat_source.shape[-1]
            )
            catdata = get_maps(cat_source, cat_batchsize)
            mappings, counts = catdata.mappings, catdata.counts
            ncats = np.array([len(m) for m in mappings])
            write_categorical(cat_source, h5file, nworkers, cat_batchsize, mappings)
            cat_meta = meta.CategoricalTarget(
                N=cat_source.shape[0],
                labels=cat_source.columns,
                nvalues=ncats,
                mappings=mappings,
                counts=counts,
            )
            write_target_metadata(cat_meta, h5file)
        else:
            log.info("Reading shapefile continuous records")
            con_source = ContinuousShpArraySource(shapefile, records, random_seed)
            con_batchsize = mb_to_points(
                batchMB, ndim_con=con_source.shape[-1], ndim_cat=0
            )
            mean, sd = get_stats(con_source, con_batchsize) if normalise else None, None
            write_continuous(con_source, h5file, nworkers, con_batchsize)
            con_meta = meta.ContinuousTarget(
                N=con_source.shape[0], labels=con_source.columns, means=mean, sds=sd
            )
            write_target_metadata(con_meta, h5file)
    log.info("Target import complete")


def write_group_data(batchMB, group_col, nworkers, h5file, random_seed, shapefile):
    group_src = CategoricalShpArraySource(shapefile, [group_col], random_seed)
    group_batchsize = mb_to_points(batchMB, ndim_con=0, ndim_cat=1, ndim_coord=0)
    group_data = get_maps(group_src, group_batchsize)
    gdata_mappings, gdata_counts = group_data.mappings, group_data.counts
    write_categorical(group_src, h5file, nworkers, group_batchsize, name="groups_data", maps=gdata_mappings)
    group_data_meta = meta.GroupDataTarget(
        N=group_src.shape[0],
        labels=group_src.columns,
        nvalues=np.array([len(m) for m in gdata_mappings]),
        mappings=gdata_mappings,
        counts=gdata_counts
    )
    write_target_metadata(group_data_meta, h5file)


if __name__ == "__main__":
    cli()
