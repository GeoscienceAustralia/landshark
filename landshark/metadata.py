"""Metadata."""

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

import os.path
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from landshark.basetypes import CategoricalType, ContinuousType
from landshark.image import ImageSpec


class PickleObj:

    _filename: Optional[str] = None

    @classmethod
    def load(cls, directory: str) -> Any:
        if not cls._filename:
            raise NotImplementedError("PickleObj must be subclassed")
        path = os.path.join(directory, cls._filename)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save(self, directory: str) -> None:
        if not self._filename:
            raise NotImplementedError("PickleObj must be subclassed")
        path = os.path.join(directory, self._filename)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class CategoricalFeature(NamedTuple):
    nvalues: int
    D: int
    mapping: np.ndarray
    counts: np.ndarray


class ContinuousFeature(NamedTuple):
    D: int
    mean: np.ndarray
    sd: np.ndarray


Feature = Union[CategoricalFeature, ContinuousFeature]


class ContinuousFeatureSet:
    def __init__(
        self,
        labels: List[str],
        missing: ContinuousType,
        stats: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:

        D = len(labels)
        if stats is None:
            self.normalised = False
            means = [None] * D
            sds = [None] * D
        else:
            self.normalised = True
            means, sds = stats

        self._missing = missing
        # hard-code that each feature has 1 band for now
        self._columns = OrderedDict(
            [
                (l, ContinuousFeature(1, np.array([m]), np.array([v])))
                for l, m, v in zip(labels, means, sds)
            ]
        )
        self._n = len(self._columns)

    @property
    def columns(self) -> OrderedDict:
        return self._columns

    @property
    def missing_value(self) -> ContinuousType:
        return self._missing

    def __len__(self) -> int:
        return self._n


class CategoricalFeatureSet:
    def __init__(
        self,
        labels: List[str],
        missing: CategoricalType,
        nvalues: np.ndarray,
        mappings: List[np.ndarray],
        counts: np.ndarray,
    ) -> None:
        self._missing = missing
        # hard-code that each feature has 1 band for now
        self._columns = OrderedDict(
            [
                (l, CategoricalFeature(n, 1, m, c))
                for l, n, m, c in zip(labels, nvalues, mappings, counts)
            ]
        )
        self._n = len(self._columns)

    @property
    def columns(self) -> OrderedDict:
        return self._columns

    @property
    def missing_value(self) -> CategoricalType:
        return self._missing

    def __len__(self) -> int:
        return self._n


class FeatureSet(PickleObj):

    _filename = "FEATURESET.bin"

    def __init__(
        self,
        continuous: Optional[ContinuousFeatureSet],
        categorical: Optional[CategoricalFeatureSet],
        image: ImageSpec,
        N: int,
        halfwidth: int,
    ) -> None:
        self.continuous = continuous
        self.categorical = categorical
        self.image = image
        self._N = N
        self.halfwidth = halfwidth

    def __len__(self) -> int:
        return self._N


class CategoricalTarget(PickleObj):

    _filename = "CATEGORICALTARGET.bin"
    dtype = CategoricalType

    def __init__(
        self,
        N: int,
        labels: np.ndarray,
        nvalues: np.ndarray,
        mappings: List[np.ndarray],
        counts: List[np.ndarray],
    ) -> None:
        self.N = N
        self.D = len(labels)
        self.nvalues = nvalues
        self.mappings = mappings
        self.counts = counts
        self.labels = labels


class ContinuousTarget(PickleObj):

    _filename = "CONTINUOUSTARGET.bin"
    dtype = ContinuousType

    def __init__(
        self, N: int, labels: np.ndarray, means: np.ndarray, sds: np.ndarray
    ) -> None:
        self.N = N
        self.D = len(labels)
        self.normalised = means is not None
        self.means = means
        self.sds = sds
        self.labels = labels


Target = Union[ContinuousTarget, CategoricalTarget]


class Training(PickleObj):

    _filename = "TRAINING.bin"

    def __init__(
        self,
        targets: Target,
        features: FeatureSet,
        nfolds: int,
        testfold: int,
        fold_counts: Dict[int, int],
    ) -> None:
        self.targets = targets
        self.features = features
        self.nfolds = nfolds
        self.testfold = testfold
        self.fold_counts = fold_counts
