"""Test kfold module."""

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

import numpy as np
import pytest

from landshark.kfold import KFolds, GroupKFolds

fold_params = [(10, 2, 5), (123456, 10, 99)]


@pytest.mark.parametrize("N,K,B", fold_params)
def test_kfolds(N, K, B):
    folds = KFolds(N, K)
    ixs = list(folds.iterator(B))
    bs = [len(b) for b in ixs]
    assert bs == [B] * (N // B) + [] if N % B == 0 else [N % B]
    ixs_flat = [i for b in ixs for i in b]
    assert len(set(ixs_flat)) == K
    assert min(ixs_flat) > 0
    assert max(ixs_flat) <= K
    assert set(folds.counts.keys()) == set(range(1, K + 1))
    assert sum(folds.counts.values()) == N


rnd = np.random.RandomState(23)
group_kfold_params = [
    (rnd.randint(5, 16, size=30), 5, 10, 12),
    (rnd.randint(5, 15, size=1000), 5, 200, 13),
    (rnd.randint(5, 25, size=3000), 12, 100, 666),
    (rnd.randint(5, 15, size=3000), 7, 66, 667),
    (rnd.randint(1, 25, size=300), 7, 100, 66),
    (rnd.choice("a b c d e f g h i".split(), size=100), 3, 10, 11),
    (rnd.choice("a b c d e f g h i".split(), size=113), 7, 10, 21)
]


@pytest.mark.parametrize("groups,K,batch_size,seed", group_kfold_params)
def test_group_kfolds(groups, K, batch_size, seed):
    folds = GroupKFolds(groups, K, seed=seed)
    ixs = list(folds.iterator(batch_size))
    bs = [len(b) for b in ixs]
    N = groups.shape[0]
    assert bs == [batch_size] * (N // batch_size) + [] if N % batch_size == 0 else [N % batch_size]
    ixs_flat = [i for b in ixs for i in b]
    assert len(set(ixs_flat)) == K
    assert min(ixs_flat) > 0
    assert max(ixs_flat) <= K
    assert set(folds.counts.keys()) == set(range(1, K+1))
    assert sum(folds.counts.values()) == N
