"""Import data from tensorflow format."""

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
from glob import glob
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from landshark.metadata import FeatureSet, Target, Training
from landshark.serialise import deserialise
from landshark.util import points_per_batch

log = logging.getLogger(__name__)


def dataset_fn(
    records: List[str],
    batchsize: int,
    features: FeatureSet,
    targets: Optional[Target] = None,
    epochs: int = 1,
    take: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> Callable[[], tf.data.TFRecordDataset]:
    """Dataset feeder."""

    def f() -> tf.data.TFRecordDataset:
        dataset = tf.data.TFRecordDataset(records, compression_type="ZLIB").repeat(
            count=epochs
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=random_seed)
        dataset = (
            dataset.take(take)
            .batch(batchsize)
            .map(lambda x: deserialise(x, features, targets))
        )
        return dataset

    return f


def get_training_meta(directory: str, train_validation: bool = False) -> Tuple[Training, List[str], List[str]]:
    """Read train/test metadata and record filenames from dir."""
    test_dir = directory if train_validation else os.path.join(directory, "testing")
    training_records = glob(os.path.join(directory, "train.*.tfrecord"))
    testing_records = glob(os.path.join(test_dir, "validation.*.tfrecord")) if train_validation else \
        glob(os.path.join(test_dir, "test.*.tfrecord"))
    log.info(f"Using test data {testing_records}")
    metadata = Training.load(directory)
    return metadata, training_records, testing_records


def get_query_meta(query_dir: str) -> Tuple[FeatureSet, List[str], int, int]:
    """Read query metadata and record filenames from dir."""
    strip_list = query_dir.rstrip("/").split("strip")[-1].split("of")
    assert len(strip_list) == 2
    strip = int(strip_list[0])
    nstrip = int(strip_list[1])
    query_metadata = FeatureSet.load(query_dir)
    query_records = glob(os.path.join(query_dir, "*.tfrecord"))
    query_records.sort()
    return query_metadata, query_records, strip, nstrip


def get_oos_query_meta(oos_dir: str) -> Tuple[Training, List[str]]:
    """Read query metadata and record filenames from dir."""
    oos_records = glob(os.path.join(oos_dir, "train.*.tfrecord"))
    oos_metadata = Training.load(oos_dir)
    return oos_metadata, oos_records


def _make_mask(
    x: Dict[str, np.ndarray], xm: Dict[str, np.ndarray]
) -> Dict[str, np.ma.MaskedArray]:
    """Combine arrays and masks to MaskedArray's."""
    assert x.keys() == xm.keys()
    d = {k: np.ma.MaskedArray(data=x[k], mask=xm[k]) for k in x.keys()}
    return d


TData = Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]


class XData(NamedTuple):
    """Container for covariate data X."""

    x_con: Optional[Dict[str, np.ma.MaskedArray]]
    x_cat: Optional[Dict[str, np.ma.MaskedArray]]
    indices: np.ndarray
    coords: np.ndarray


class XYData(NamedTuple):
    """Container for covariate X and target data Y."""

    x_con: Optional[Dict[str, np.ma.MaskedArray]]
    x_cat: Optional[Dict[str, np.ma.MaskedArray]]
    indices: np.ndarray
    coords: np.ndarray
    y: Dict[str, np.ndarray]


def _split(X: TData) -> XData:
    """Split dict into elements."""
    Xcon = _make_mask(X["con"], X["con_mask"]) if "con" in X else None
    Xcat = _make_mask(X["cat"], X["cat_mask"]) if "cat" in X else None
    return XData(Xcon, Xcat, X["indices"], X["coords"])


def _concat_dict(xlist: List[TData]) -> TData:
    """Join dicts of arrays together."""
    out_dict = {}
    for k, v in xlist[0].items():
        if tf.is_tensor(v):
            out_dict[k] = np.concatenate([di[k] for di in xlist], axis=0)
        else:
            out_dict[k] = _concat_dict([di[k] for di in xlist])
    return out_dict


def extract_split_xy(dataset: tf.data.TFRecordDataset) -> XYData:
    """Extract (X, Y) data from tensor dataset and split."""
    x_list, y_list = zip(*dataset)
    Y = np.concatenate(y_list, axis=0)
    X = _concat_dict(x_list)
    x_con, x_cat, indices, coords = _split(X)

    return XYData(x_con, x_cat, indices, coords, Y)


def xy_record_data(
    records: List[str],
    metadata: Training,
    batchsize: int = 1000,
    npoints: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> XYData:
    """Read train/test record."""
    train_dataset = dataset_fn(
        records=records,
        batchsize=batchsize,
        features=metadata.features,
        targets=metadata.targets,
        take=npoints,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        random_seed=random_seed,
    )()
    xy_data_tuple = extract_split_xy(train_dataset)
    return xy_data_tuple


def query_data_it(
    records: List[str],
    features: FeatureSet,
    batchsize: int,
    npoints: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> Iterator[XData]:
    """Exctract query data from tfrecord in batches."""
    dataset = dataset_fn(
        records=records,
        batchsize=batchsize,
        features=features,
        take=npoints,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        random_seed=random_seed,
    )()
    for X in dataset:
        yield _split(X)


#
# functions for inspecting tfrecord data directly
#


def read_train_record(
    directory: str,
    npoints: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> XYData:
    """Read train record."""
    metadata, train_records, _ = get_training_meta(directory)
    train_data_tuple = xy_record_data(
        records=train_records,
        metadata=metadata,
        npoints=npoints,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        random_seed=random_seed,
    )
    return train_data_tuple


def read_test_record(
    directory: str,
    npoints: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> XYData:
    """Read test record."""
    metadata, _, test_records = get_training_meta(directory)
    test_data_tuple = xy_record_data(
        records=test_records,
        metadata=metadata,
        npoints=npoints,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        random_seed=random_seed,
    )
    return test_data_tuple


def read_query_record(
    query_dir: str,
    batch_mb: float,
    npoints: int = -1,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    random_seed: Optional[int] = None,
) -> Iterator[XData]:
    """Read query data in batches."""
    features, records, strip, nstrip = get_query_meta(query_dir)
    batchsize = points_per_batch(features, batch_mb)
    yield from query_data_it(
        records=records,
        features=features,
        batchsize=batchsize,
        npoints=npoints,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        random_seed=random_seed,
    )
