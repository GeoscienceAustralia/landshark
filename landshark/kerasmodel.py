"""Train/test with tfrecords."""

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
import signal
from itertools import count
from typing import (Any, Callable, Dict, Generator, List, NamedTuple, Optional,
                    Sequence, Tuple)

import numpy as np
import tensorflow as tf

from landshark.metadata import Training
from landshark.model import (QueryConfig, TrainingConfig, predict_data,
                             test_data, train_data)
from landshark.saver import BestScoreSaver
from landshark.serialise import deserialise

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


class FeatInput(NamedTuple):

    data: tf.keras.Input
    mask: Optional[tf.keras.Input]


def impute_const_fn(x: Sequence[tf.Tensor], value: int = 0) -> tf.Tensor:
    data, mask = x
    tmask = tf.cast(mask, dtype=data.dtype)
    fmask = tf.cast(tf.logical_not(mask), dtype=data.dtype)
    data_imputed = data * fmask + value * tmask
    return data_imputed


def impute_const_layer(feat: FeatInput, value: int = 0) -> tf.keras.layers.Layer:
    if feat.mask is not None:
        layer = tf.keras.layers.Lambda(impute_const_fn, value)(feat)
    else:
        layer = tf.keras.layers.InputLayer(feat.data)
    return layer


def get_feat_input_list(
    num_feats: List[FeatInput],
    cat_feats: List[Tuple[FeatInput, int]]
) -> List[tf.keras.Input]:
    num = [x for f in num_feats for x in f]
    cat = [x for f, _ in cat_feats for x in f if x is not None]
    return num + cat


class KerasInputs(NamedTuple):

    num_feats: List[FeatInput]
    cat_feats: List[Tuple[FeatInput, int]]
    indices: tf.keras.Input
    coords: tf.keras.Input


def gen_keras_inputs(
    dataset: tf.data.TFRecordDataset,
    metadata: Training,
) -> KerasInputs:
    xs, _ = dataset.element_spec

    def gen_feat_input(
        data: tf.TensorSpec, mask: tf.TensorSpec, name: str
    ) -> FeatInput:
        feat = FeatInput(
            data=tf.keras.Input(name=name, shape=data.shape[1:], dtype=data.dtype),
            mask=tf.keras.Input(name=f"{name}_mask", shape=mask.shape[1:], dtype=mask.dtype)
        )
        return feat

    num_feats = [
        gen_feat_input(xs["con"][k], xs["con_mask"][k], k)
        for k in xs.get("con", [])
    ]
    if "cat" in xs:
        assert "cat_mask" in xs and metadata.features.categorical
        cat_feats = [
            (
                gen_feat_input(xs["cat"][k], xs["cat_mask"][k], k),
                metadata.features.categorical.columns[k].mapping.shape[0]
            )
            for k in xs["cat"]
        ]

    feats = KerasInputs(
        num_feats=num_feats,
        cat_feats=cat_feats,
        indices=tf.keras.Input(name="indices", shape=xs["indices"].shape),
        coords=tf.keras.Input(name="coords", shape=xs["coords"].shape),
    )
    return feats


def flatten_dataset(d, ignore_y=False):
    print(d)
    #d = d if ignore_y else d[0]

    def _flat_mask(x, key):
        x_flat = {
            **x.get(key, {}),
            **{f"{k}_mask": v for k, v in x.get(f"{key}_mask", {}).items()},
        }
        return x_flat

    d_flat = {**_flat_mask(d, "con"), **_flat_mask(d, "cat")}
    return d_flat


def train_test(records_train: List[str],
               records_test: List[str],
               metadata: Training,
               directory: str,
               cf: Any,  # Module type
               params: TrainingConfig,
               iterations: Optional[int]
               ) -> None:
    """Model training and periodic hold-out testing."""
    xtrain = train_data(records_train, metadata, params.batchsize,
                        params.epochs)()
    xtest = test_data(records_test, metadata, params.test_batchsize)()

    inputs = gen_keras_inputs(xtrain, metadata)

    model = cf.model(*inputs, metadata.targets.D)

    xtrain = xtrain.map(flatten_dataset)
    xtest = xtest.map(flatten_dataset)

    model.fit(
        x=xtrain,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=xtest,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    a = 1
    return


def predict(checkpoint_dir: str,
            model: Any,  # Module type
            metadata: Training,
            records: List[str],
            params: QueryConfig
            ) -> Generator:
    """Load a model and predict results for record inputs."""
    x = predict_data(records, metadata, params.batchsize)

    it = model.predict(
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    yield it
