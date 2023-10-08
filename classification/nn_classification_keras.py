"""Example regression config using a Keras Model."""

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

from typing import List

import tensorflow as tf

from landshark.kerasmodel import (
    CatFeatInput,
    NumFeatInput,
    TargetData,
    get_feat_input_list,
    impute_embed_concat_layer,
)
# from landshark.metadata import Training
from landshark.model import TrainingConfig, QueryConfig, Training

def model(
    num_feats: List[NumFeatInput],
    cat_feats: List[CatFeatInput],
    indices: tf.keras.Input,
    coords: tf.keras.Input,
    targets: List[TargetData],
    metadata: Training,
    training_params: TrainingConfig
) -> tf.keras.Model:
    """Example model config.
    Must match the signature above and return a compiled tf.keras.Model
    """

    l0 = impute_embed_concat_layer(num_feats, cat_feats, cat_embed_dims=3)

    if metadata.features.halfwidth > 0:
        # Conv2D NN
        l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation=tf.nn.relu)(l0)
        l2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation=tf.nn.relu)(l1)
    else:
        # dense NN
        l1 = tf.keras.layers.Dense(units=64, activation="relu")(l0)
        l2 = tf.keras.layers.Dense(units=32, activation="relu")(l1)

    # Get some predictions for the labels
    n_outputs = sum(t.n_classes for t in targets)
    l3 = tf.keras.layers.Dense(units=n_outputs, activation=None)(l2)

    outputs = []
    losses = {}
    metrics = {}
    i = 0
    for t in targets:
        target_vals = l3[..., i: i + t.n_classes]
        logits = tf.keras.layers.Reshape((-1,), name=t.label)(target_vals)
        losses[t.label] = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        metrics[t.label] = "accuracy"

        pred_name = f"predictions_most_probable_{t.label}"
        pred = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.cast(tf.keras.backend.argmax(x), "uint8"),
            name=pred_name,
        )(logits)
        outputs.extend([logits, pred])
        probabilities = tf.keras.layers.Softmax(dtype="float32")(logits)
        class_probabilities = [
            tf.keras.layers.Lambda(
                lambda x, jj: x[:, jj],
                name=f"predictions_probability_class_{j}",
                dtype="float32",
                arguments={"jj": j}
            )(probabilities) for j in range(t.n_classes)
        ]
        outputs.extend(class_probabilities)
        i += t.n_classes

    # create keras model
    model_inputs = get_feat_input_list(num_feats, cat_feats)
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model
