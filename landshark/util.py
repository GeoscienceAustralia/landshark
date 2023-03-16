"""Utilities."""

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
from typing import Tuple
from collections import defaultdict

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

from landshark.basetypes import (
    CategoricalType,
    ContinuousType,
    CoordinateType,
    MissingType,
)
from landshark.metadata import FeatureSet

log = logging.getLogger(__name__)


def lins_ccc(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient.

    See https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    Parameters
    ----------
    y_true: ndarray
        vector of true targets
    y_pred: ndarray
        vector of predicted targets

    Returns
    -------
    float:
        1.0 for a perfect match between :code:`y_true` and :code:`y_pred`, less
        otherwise

    Example
    -------
    >>> y_true = np.random.randn(100)
    >>> lins_ccc(y_true, y_true) > 0.99  # Should be good predictor
    True
    >>> lins_ccc(y_true, np.zeros_like(y_true)) < 0.01  # Bad predictor
    True
    """

    t = y_true.mean()
    p = y_pred.mean()
    St = y_true.var()
    Sp = y_pred.var()
    Spt = np.mean((y_true - t) * (y_pred - p))

    return 2 * Spt / (St + Sp + (t - p)**2)


SCORES = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "ex_var": explained_variance_score,
    'lins_ccc': lins_ccc
}


def score(labels, y_true, y_pred):
    scores = defaultdict(dict)
    for i, l in enumerate(labels):
        if not l.endswith("_std"):
            for s, func in SCORES.items():
                scores[l][s] = func(y_true[:, 0], y_pred[:, i])
    return scores


def to_masked(array: np.ndarray, missing_value: MissingType) -> np.ma.MaskedArray:
    """Create a masked array from array plus list of missing."""
    if missing_value is None:
        marray = np.ma.MaskedArray(data=array, mask=np.ma.nomask)
    else:
        mask = array == missing_value
        marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


def _batch_points(
    batchMB: float,
    ndim_con: int,
    ndim_cat: int,
    ndim_coord: int = 0,
    halfwidth: int = 0,
) -> Tuple[float, float]:
    patchsize = (halfwidth * 2 + 1) ** 2
    bytes_con = np.dtype(ContinuousType).itemsize * ndim_con
    bytes_cat = np.dtype(CategoricalType).itemsize * ndim_cat
    bytes_coord = np.dtype(CoordinateType).itemsize * ndim_coord
    mbytes_per_point = (bytes_con + bytes_cat + bytes_coord) * patchsize * 1e-6
    npoints = batchMB / mbytes_per_point
    return npoints, mbytes_per_point


def mb_to_points(
    batchMB: float,
    ndim_con: int,
    ndim_cat: int,
    ndim_coord: int = 0,
    halfwidth: int = 0,
) -> int:
    """Calculate the number of points of data to fill a memory allocation."""
    log.info("Batch size of {}MB requested".format(batchMB))
    npoints, mb_per_point = _batch_points(
        batchMB, ndim_con, ndim_cat, ndim_coord, halfwidth
    )
    npoints = int(round(max(1.0, npoints)))
    log.info(
        "Batch size set to {} points, total {:0.2f}MB".format(
            npoints, npoints * mb_per_point
        )
    )
    return npoints


def mb_to_rows(batchMB: float, row_width: int, ndim_con: int, ndim_cat: int) -> int:
    """Calculate the number of rows of data to fill a memory allocation."""
    log.info("Batch size of {}MB requested".format(batchMB))
    npoints, mb_per_point = _batch_points(batchMB, ndim_con, ndim_cat)
    nrows = int(round(max(1.0, npoints / row_width)))
    log.info(
        "Batch size set to {} rows, total {:0.2f}MB".format(
            nrows, mb_per_point * row_width * nrows
        )
    )
    return nrows


def points_per_batch(meta: FeatureSet, batch_mb: float) -> int:
    """Calculate batchsize in points given a memory allocation."""
    ndim_con = len(meta.continuous.columns) if meta.continuous else 0
    ndim_cat = len(meta.categorical.columns) if meta.categorical else 0
    batchsize = mb_to_points(batch_mb, ndim_con, ndim_cat, halfwidth=meta.halfwidth)
    return batchsize
