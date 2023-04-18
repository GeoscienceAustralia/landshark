"""High-level file IO operations and utility functions."""

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
import csv
from glob import glob
from typing import List, Optional


def tifnames(
    directories: Optional[List[str]] = None,
    list_files: Optional[List[str]] = None
    ) -> List[str]:
    """Recursively find all tif/gtif files within a list of directories."""
    names: List[str] = []
    assert (directories is not None) or (list_files is not None)  # both can't be null
    directories = [] if directories is None else directories
    list_files = [] if list_files is None else list_files

    for d in directories:
        file_types = ("tif", "gtif")
        for t in file_types:
            if os.path.isfile(d) and d.endswith(f".{t}"):
                names.append(os.path.abspath(d))
                break
            glob_pattern = os.path.join(d, "**", "*.{}".format(t))
            files = glob(glob_pattern, recursive=True)
            names.extend([os.path.abspath(f) for f in files])

    for l in list_files:
        csvfile = os.path.abspath(l)
        files = []
        with open(csvfile, 'r') as f:
            reader = csv.reader(f)
            tifs = list(reader)
            tifs = [f[0].strip() for f in tifs
                    if (len(f) > 0 and f[0].strip() and
                        f[0].strip()[0] != '#')]
        for f in tifs:
            files.append(os.path.abspath(f))

        names.extend(sorted(files, key=str.lower))
    return list(set(names))
