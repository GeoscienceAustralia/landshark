"""Import tifs and targets into landshark world."""

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
import os
from multiprocessing import cpu_count
from typing import NamedTuple, Tuple

import click

from landshark import __version__, errors
from landshark import metadata as meta
from landshark.dataprocess import (
    ProcessQueryArgs,
    ProcessTrainingArgs,
    ProcessTrainingValidationArgs,
    write_querydata,
    write_trainingdata,
    write_training_validation_data,
)
from landshark.featurewrite import read_feature_metadata, read_target_metadata, read_groups_data_metadata
from landshark.hread import CategoricalH5ArraySource, ContinuousH5ArraySource, GroupsH5ArraySource, H5TargetGroupShape
from landshark.image import strip_image_spec
from landshark.kfold import KFolds, GroupKFolds
from landshark.scripts.logger import configure_logging
from landshark.util import points_per_batch

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
def cli(ctx: click.Context, verbosity: str, batch_mb: float, nworkers: int) -> int:
    """Extract features and targets for training, testing and prediction."""
    ctx.obj = CliArgs(nworkers, batch_mb)
    configure_logging(verbosity)
    return 0


@cli.command()
@click.option(
    "--targets",
    type=click.Path(exists=True),
    required=True,
    help="Target HDF5 file from which to read",
)
@click.option(
    "--kfold/--group_kfold",
    is_flag=True,
    default=True,
    help="Use group kfold",
)
@click.option(
    "--split",
    type=int,
    nargs=2,
    default=(1, 10),
    help="Train/test split fold structure. First argument is test "
    "fold (counting from 1), second is total folds.",
)
@click.option(
    "--random_seed",
    type=int,
    default=666,
    help="Random state for assigning data to folds",
)
@click.option("--name", type=str, required=True, help="Name of the output folder")
@click.option(
    "--features",
    type=click.Path(exists=True),
    required=True,
    help="Feature HDF5 file from which to read",
)
@click.option(
    "--halfwidth",
    type=int,
    default=0,
    help="half width of patch size. Patch side length is " "2 x halfwidth + 1",
)
@click.pass_context
def traintest(
    ctx: click.Context,
    targets: str,
    kfold: bool,
    split: Tuple[int, ...],
    random_seed: int,
    name: str,
    features: str,
    halfwidth: int,
) -> None:
    """Extract training and testing data to train and validate a model."""
    fold, nfolds = split
    catching_f = errors.catch_and_exit(traintest_entrypoint)
    catching_f(
        targets,
        kfold,
        fold,
        nfolds,
        random_seed,
        name,
        halfwidth,
        ctx.obj.nworkers,
        features,
        ctx.obj.batchMB,
    )


def traintest_entrypoint(
    targets: str,
    kfold: bool,
    testfold: int,
    folds: int,
    random_seed: int,
    name: str,
    halfwidth: int,
    nworkers: int,
    features: str,
    batchMB: float,
) -> None:
    """Get training data."""
    feature_metadata = read_feature_metadata(features)
    feature_metadata.halfwidth = halfwidth
    target_metadata = read_target_metadata(targets)
    batchsize = points_per_batch(feature_metadata, batchMB)
    target_src = (
        CategoricalH5ArraySource(targets)
        if isinstance(target_metadata, meta.CategoricalTarget)
        else ContinuousH5ArraySource(targets)
    )
    if not kfold:
        group_meta = read_groups_data_metadata(targets)
        group_src = GroupsH5ArraySource(targets)
        assert testfold <= folds
        assert folds <= group_meta.N
        target_groups = H5TargetGroupShape(targets)
        kfolds = GroupKFolds(target_groups.groups, folds, random_seed)
    else:
        n_rows = len(target_src)
        kfolds = KFolds(n_rows, folds, random_seed)

    directory = os.path.join(
        os.getcwd(), "traintest_{}_fold{}of{}".format(name, testfold, folds)
    )

    args = ProcessTrainingArgs(
        name=name,
        feature_path=features,
        target_src=target_src,
        image_spec=feature_metadata.image,
        halfwidth=halfwidth,
        testfold=testfold,
        folds=kfolds,
        directory=directory,
        batchsize=batchsize,
        nworkers=nworkers,
        tag='train',
    )
    write_trainingdata(args)
    training_metadata = meta.Training(
        targets=target_metadata,
        features=feature_metadata,
        nfolds=folds,
        testfold=testfold,
        fold_counts=kfolds.counts,
    )
    training_metadata.save(directory)
    log.info("Training import complete")


@cli.command()
@click.option(
    "--targets",
    type=click.Path(exists=True),
    required=True,
    help="Target HDF5 file from which to read",
)
@click.option(
    "--validation",
    type=click.Path(exists=True),
    required=True,
    help="Validation HDF5 file from which to read",
)
@click.option(
    "--random_seed",
    type=int,
    default=666,
    help="Random state for assigning data to folds",
)
@click.option("--name", type=str, required=True, help="Name of the output folder")
@click.option(
    "--features",
    type=click.Path(exists=True),
    required=True,
    help="Feature HDF5 file from which to read",
)
@click.option(
    "--halfwidth",
    type=int,
    default=0,
    help="half width of patch size. Patch side length is " "2 x halfwidth + 1",
)
@click.pass_context
def trainvalidate(
        ctx: click.Context,
        targets: str,
        validation: str,
        random_seed: int,
        name: str,
        features: str,
        halfwidth: int,
) -> None:
    """Extract training and testing data to train and validate a model."""
    # fold, nfolds = split
    catching_f = errors.catch_and_exit(trainvalidate_entrypoint)
    catching_f(
        targets,
        validation,
        random_seed,
        name,
        halfwidth,
        ctx.obj.nworkers,
        features,
        ctx.obj.batchMB,
    )


def trainvalidate_entrypoint(
        targets: str,
        validation: str,
        random_seed: int,
        name: str,
        halfwidth: int,
        nworkers: int,
        features: str,
        batchMB: float,
) -> None:
    """Get training data."""
    feature_metadata = read_feature_metadata(features)
    feature_metadata.halfwidth = halfwidth
    target_metadata = read_target_metadata(targets)
    validation_metadata = read_target_metadata(validation)
    batchsize = points_per_batch(feature_metadata, batchMB)
    target_src = (
        CategoricalH5ArraySource(targets)
        if isinstance(target_metadata, meta.CategoricalTarget)
        else ContinuousH5ArraySource(targets)
    )

    validation_src = (
        CategoricalH5ArraySource(validation)
        if isinstance(validation_metadata, meta.CategoricalTarget)
        else ContinuousH5ArraySource(validation)
    )

    kfolds = KFolds(len(target_src), 10, random_seed)
    vkfolds = KFolds(len(validation_src), 10, random_seed)

    directory = os.path.join(
        os.getcwd(), "trainvalidate_{}".format(name, targets)
    )
    args = ProcessTrainingValidationArgs(
        training_args=ProcessTrainingArgs(
            name=name,
            feature_path=features,
            target_src=target_src,
            image_spec=feature_metadata.image,
            halfwidth=halfwidth,
            directory=directory,
            batchsize=batchsize,
            nworkers=nworkers,
            testfold=-1,  # no test fold, the whole shapefile is training set
            folds=kfolds,
            tag='train',
          ),
        validation_args=ProcessTrainingArgs(
            name=name,
            feature_path=features,
            target_src=validation_src,
            image_spec=feature_metadata.image,
            halfwidth=halfwidth,
            directory=directory,
            batchsize=batchsize,
            nworkers=nworkers,
            testfold=-1,  # no test fold, the entire shapefile is validation set
            folds=vkfolds,
            tag='validation',
        )
    )

    write_training_validation_data(args)
    training_val_metadata = meta.TrainingValidate(
        targets=target_metadata,
        validation=validation_metadata,
        features=feature_metadata,
    )
    training_val_metadata.save(directory)
    log.info("Training import complete")


@cli.command()
@click.option(
    "--strip",
    type=int,
    nargs=2,
    default=(1, 1),
    help="Horizontal strip of the image, eg --strip 3 5 is the " "third strip of 5",
)
@click.option(
    "--name", type=str, required=True, help="The name of the output from this command."
)
@click.option(
    "--features",
    type=click.Path(exists=True),
    required=True,
    help="Feature HDF5 file from which to read",
)
@click.option(
    "--halfwidth",
    type=int,
    default=0,
    help="half width of patch size. Patch side length is " "2 x halfwidth + 1",
)
@click.option(
    "--remote",
    type=str,
    default='',
    help="remote dir location. For example in nci could be '/jobfs/'",
)
@click.pass_context
def query(
    ctx: click.Context, strip: Tuple[int, int], name: str, features: str, halfwidth: int, remote: str
) -> None:
    """Extract query data for making prediction images."""
    catching_f = errors.catch_and_exit(query_entrypoint)
    catching_f(features, ctx.obj.batchMB, ctx.obj.nworkers, halfwidth, strip, name, remote)


def query_entrypoint(
    features: str,
    batchMB: float,
    nworkers: int,
    halfwidth: int,
    strip: Tuple[int, int],
    name: str,
    remote_dir_location: str,
) -> int:
    """Entrypoint for extracting query data."""
    strip_idx, totalstrips = strip
    assert strip_idx > 0 and strip_idx <= totalstrips

    """Grab a chunk for prediction."""
    log.info("Using {} worker processes".format(nworkers))

    dirname = "{}query_{}_strip{}of{}".format(remote_dir_location, name, strip_idx, totalstrips)
    if remote_dir_location == '':
        directory = os.path.join(os.getcwd(), dirname)
    else:
        directory = dirname
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    feature_metadata = read_feature_metadata(features)
    feature_metadata.halfwidth = halfwidth
    batchsize = points_per_batch(feature_metadata, batchMB)
    strip_imspec = strip_image_spec(strip_idx, totalstrips, feature_metadata.image)
    tag = "query.{}of{}".format(strip_idx, totalstrips)

    qargs = ProcessQueryArgs(
        name,
        features,
        feature_metadata.image,
        strip_idx,
        totalstrips,
        strip_imspec,
        halfwidth,
        directory,
        batchsize,
        nworkers,
        tag,
    )

    write_querydata(qargs)
    feature_metadata.image = strip_imspec
    feature_metadata.save(directory)
    log.info("Query import complete")
    return 0


if __name__ == "__main__":
    cli()
