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
import tensorflow_probability as tfp

from landshark.kerasmodel import (
    CatFeatInput,
    NumFeatInput,
    TargetData,
    get_feat_input_list,
    impute_embed_concat_layer,
)
from landshark.metadata import Training


def r2(y_true, y_pred):
    """Coefficient of determination metric."""
    SS_res = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))
    SS_tot = tf.reduce_sum(tf.math.squared_difference(y_true, tf.reduce_mean(y_true)))
    return 1 - SS_res / SS_tot


def orig_model(
        num_feats: List[NumFeatInput],
        cat_feats: List[CatFeatInput],
        indices: tf.keras.Input,
        coords: tf.keras.Input,
        targets: List[TargetData],
        metadata: Training,
) -> tf.keras.Model:
    """Example model config.
    Must match the signature above and return a compiled tf.keras.Model
    """

    l0 = impute_embed_concat_layer(num_feats, cat_feats, cat_embed_dims=3)

    # Conv2D NN
    if metadata.features.halfwidth > 0:
        l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation=tf.nn.relu)(l0)
        l = tf.keras.layers.Dropout(0.1)(l1)
        l = tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                                   activation=tf.nn.relu)(l)
        l = tf.keras.layers.Dropout(0.1)(l)
        # l = tf.keras.layers.Conv2D(filters=16, kernel_size=2,
        #                            activation=tf.nn.relu)(l)
        #l = tf.keras.layers.Dropout(0.1)(l)
        #l = tf.keras.layers.Conv2D(filters=16, kernel_size=2,
        #                            activation=tf.nn.relu)(l)
        l2 = tf.keras.layers.Dense(units=16, activation="relu")(l)
    else:
        l1 = tf.keras.layers.Dense(units=256, activation="relu")(l0)
        l11 = tf.keras.layers.Dropout(0.1)(l1)
        l21 = tf.keras.layers.Dense(units=128, activation="relu")(l11)
        l22 = tf.keras.layers.Dropout(0.1)(l21)
        l23 = tf.keras.layers.Dense(units=64, activation="relu")(l22)
        l24 = tf.keras.layers.Dropout(0.1)(l23)
        l25 = tf.keras.layers.Dense(units=32, activation="relu")(l24)
        l26 = tf.keras.layers.Dropout(0.1)(l25)
        l2 = tf.keras.layers.Dense(units=16, activation="relu")(l26)

    flat = tf.keras.layers.Flatten()(l2)
    ys = []
    for i, t in enumerate(targets):
        out = tf.keras.layers.Dense(40)(flat)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = tf.keras.layers.Dense(20)(out)
        out = tf.keras.layers.Dense(1, name=f"predictions_{t.label}")(out)
        ys.append(out)

    # Get some predictions for the labels
    # create keras model
    model_inputs = get_feat_input_list(num_feats, cat_feats)
    model = tf.keras.Model(inputs=model_inputs, outputs=ys)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=[r2])
    model.summary()
    return model


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def negative_loglikelihood(target_obs, estimated_distribution):
    # tf.print("SHAPE:", target_obs.shape)
    tf.print("OBS_5:", target_obs[:5, 0])
    # tf.print("LEN_:", len(target_obs[:, 0]))
    tf.print("XYZ_:", -estimated_distribution.log_prob(target_obs)[:5])
    return -estimated_distribution.log_prob(target_obs)


def model(
        num_feats: List[NumFeatInput],
        cat_feats: List[CatFeatInput],
        indices: tf.keras.Input,
        coords: tf.keras.Input,
        targets: List[TargetData],
        metadata: Training,
) -> tf.keras.Model:
    """Example model config.
    Must match the signature above and return a compiled tf.keras.Model
    """

    l0 = impute_embed_concat_layer(num_feats, cat_feats, cat_embed_dims=3)

    # l1 = tf.keras.layers.Dense(units=256, activation="relu")(l0)
    # l11 = tf.keras.layers.Dropout(0.1)(l1)
    # l21 = tf.keras.layers.Dense(units=128, activation="relu")(l11)
    # l22 = tf.keras.layers.Dropout(0.1)(l21)
    # l23 = tf.keras.layers.Dense(units=64, activation="relu")(l22)
    # l24 = tf.keras.layers.Dropout(0.1)(l23)
    # l25 = tf.keras.layers.Dense(units=32, activation="relu")(l24)
    # l26 = tf.keras.layers.Dropout(0.1)(l25)
    # l2 = tf.keras.layers.Dense(units=16, activation="relu")(l26)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    l1 = tfp.layers.DenseVariational(
        # units=64,
        units=16,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / 51920,
        # kl_weight=1 / 236009650,
        activation="sigmoid"
    )(l0)
    l2 = tfp.layers.DenseVariational(
        # units=32,
        units=8,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / 51920,
        # kl_weight=1 / 236009650,
        activation="sigmoid"
    )(l1)

    # n_targets = len(targets)
    # l3 = tf.keras.layers.Dense(units=2, activation=None)(l2)
    # flat = tf.keras.layers.Flatten()(l3)
    ys = []
    for i, t in enumerate(targets):
        l3 = tf.keras.layers.Dense(units=2, activation=None)(l2)
        flat = tf.keras.layers.Flatten()(l3)
        out = tfp.layers.IndependentNormal(1)(flat)
        ys.append(out)

    # l3 = tf.keras.layers.Dense(units=2*n_targets, activation=None)(l2)
    # flat = tf.keras.layers.Flatten()(l3)
    # l4 = tfp.layers.IndependentNormal(n_targets)(flat)
    model_inputs = get_feat_input_list(num_feats, cat_feats)
    # model = tf.keras.Model(inputs=model_inputs, outputs={"predictions": l4})
    model = tf.keras.Model(inputs=model_inputs, outputs=ys)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss=negative_loglikelihood, optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model