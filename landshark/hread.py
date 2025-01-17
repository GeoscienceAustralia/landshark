"""Read features and targets from HDF5 files."""

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

from types import TracebackType
from typing import Tuple, Union

import numpy as np
import tables

from landshark.basetypes import (
    ArraySource,
    CategoricalArraySource,
    ContinuousArraySource,
)
from landshark.featurewrite import read_feature_metadata, read_target_metadata, read_groups_data_metadata


class H5ArraySource(ArraySource):
    """HDF5 array source for target data.

    Note these are only used for targets! see the target specific metadata
    call. Should probably be renamed.
    """

    _array_name = ""

    def __init__(self, path: str) -> None:
        self._path = path
        self.metadata = read_target_metadata(path)
        with tables.open_file(self._path, "r") as hfile:
            carray = hfile.get_node("/" + self._array_name)
            self._shape = tuple(list(carray.shape) + [carray.atom.dtype.shape[0]])
            self._missing = carray.attrs.missing
            self._native = carray.chunkshape[0]
            self._dtype = carray.atom.dtype.base

    def __enter__(self) -> None:
        self._hfile = tables.open_file(self._path, "r")
        self._carray = self._hfile.get_node("/" + self._array_name)
        if hasattr(self._hfile.root, "coordinates"):
            self._coords = self._hfile.root.coordinates
        super().__enter__()

    def __exit__(self, ex_type: type, ex_val: Exception, ex_tb: TracebackType) -> None:
        self._hfile.close()
        del self._carray
        if hasattr(self, "_coords"):
            del self._coords
        del self._hfile
        super().__exit__(ex_type, ex_val, ex_tb)

    def _arrayslice(
        self, start: int, end: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        # TODO: Note this is bad because I'm changing the return type.

        data = self._carray[start:end]
        if hasattr(self, "_coords"):
            coords = self._coords[start:end]
            return data, coords
        else:
            return data


class ContinuousH5ArraySource(H5ArraySource, ContinuousArraySource):
    """HDF5 array source for continuous target data."""

    _array_name = "continuous_data"


class CategoricalH5ArraySource(H5ArraySource, CategoricalArraySource):
    """HDF5 array source for categorical target data."""

    _array_name = "categorical_data"


class GroupsH5ArraySource(H5ArraySource, CategoricalArraySource):
    """HDF5 array source for groups data."""

    _array_name = "groups_data"


class H5Features:
    """Note unlike the array classes this isn't picklable."""

    def __init__(self, h5file: str) -> None:

        self.continuous, self.categorical, self.coordinates = None, None, None
        self.metadata = read_feature_metadata(h5file)
        self._hfile = tables.open_file(h5file, "r")
        if hasattr(self._hfile.root, "continuous_data"):
            self.continuous = self._hfile.root.continuous_data
            assert self.metadata.continuous is not None
            self.continuous.missing = self.metadata.continuous.missing_value
        if hasattr(self._hfile.root, "categorical_data"):
            self.categorical = self._hfile.root.categorical_data
            assert self.metadata.categorical is not None
            self.categorical.missing = self.metadata.categorical.missing_value
        if self.continuous:
            self._n = len(self.continuous)
        if self.categorical:
            self._n = len(self.categorical)
        if self.continuous and self.categorical:
            assert len(self.continuous) == len(self.categorical)

    def __len__(self) -> int:
        return self._n

    def __del__(self) -> None:
        self._hfile.close()


class H5TargetGroupShape:
    """
    HDF5 source for groups data
    Note unlike the array classes this isn't picklable.
    """

    def __init__(self, h5file: str) -> None:

        self.metadata = read_groups_data_metadata(h5file)
        self._hfile = tables.open_file(h5file, "r")
        self._n = self.metadata.N

        if hasattr(self._hfile.root, "groups_data"):
            mapped_groups = np.array(self._hfile.root.groups_data.read(), dtype=int)
        else:
            raise RuntimeError(f"Can't find required groups_data section in {h5file}")

        self.groups = self.metadata.mappings[0][mapped_groups]

    def __len__(self) -> int:
        return self._n

    def __del__(self) -> None:
        self._hfile.close()
