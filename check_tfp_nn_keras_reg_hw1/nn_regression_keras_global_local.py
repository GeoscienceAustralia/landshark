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

from typing import List, Union

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk

from landshark.kerasmodel import (
    CatFeatInput,
    NumFeatInput,
    TargetData,
    get_feat_input_list,
    impute_embed_concat_layer,
)
from landshark.metadata import Training
from landshark.model import TrainingConfig, QueryConfig

tfd = tfp.distributions


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
                lambda t: tfd.MultivariateNormalDiag(
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


def get_varlayer(lay, units, kl_weight):
    """ a convenience func to generate a DenseVariational layer"""
    return tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=kl_weight,
            activation="sigmoid"
            )(lay)


def get_var_flipout_layer(lay, units, k_div_fn):
    """ a convenience func to generate a DenseVariational layer"""
    return tfp.layers.DenseFlipout(
        units=units,
        kernel_divergence_fn=k_div_fn,
        activation="sigmoid"
    )(lay)


def get_convo(lay, filters, k_size, k_div_fn, padding):
    """ a convenience func to generate a convolutional probabilistic layer"""
    return tfp.layers.Convolution2DFlipout(
            filters=filters,
            kernel_size=k_size,
            kernel_divergence_fn=k_div_fn,
            padding=padding,
            # activation=tf.nn.sigmoid)(lay)
            activation=tf.nn.relu)(lay)


def negative_loglikelihood(target_obs, estimated_distribution):
    return -estimated_distribution.log_prob(target_obs)


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

    training_batchsize = training_params.batchsize
    kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(training_batchsize, dtype=tf.float32)

    def get_distlayer(lay, name=None):
        """ a helper func"""
        ln = tf.keras.layers.Dense(units=2, activation=None)(lay)
        flat = tf.keras.layers.Flatten()(ln)
        return tfp.layers.IndependentNormal(1, name=name)(flat)

    def get_flipoutlayer(lay):
        """ a helper func"""
        flat = tf.keras.layers.Flatten()(lay)
        ln = tfp.layers.DenseFlipout(8, activation=tf.nn.relu)(flat)
        return tfp.layers.DenseFlipout(1)(ln)

    model_inputs = get_feat_input_list(num_feats, cat_feats)
    l0 = impute_embed_concat_layer(num_feats, cat_feats, cat_embed_dims=3)

    if metadata.features.halfwidth == 0:
        # Create hidden layers with weight uncertainty using the DenseVariational layer.
        l1 = get_varlayer(l0, 16, 1/training_batchsize)
        l2 = get_varlayer(l1, 8, 1/training_batchsize)
        ys = [get_distlayer(l2, name=f"independent_normal_{t.label}") for t in targets]
    elif metadata.features.halfwidth == 1:
        l1 = get_convo(l0, 64, 2, kl_divergence_function, "SAME")
        l1a = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(l1)
        l2 = get_convo(l1a, 32, 2, kl_divergence_function, "SAME")
        l2a = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(l2)
        l3 = get_varlayer(l2a, 16, 1/training_batchsize)
        # ys = [get_flipoutlayer(l3) for _ in targets]
        ys = [get_distlayer(l3, name=f"independent_normal_{t.label}") for t in targets]
    else:
        # a hybrid convo-bayesian
        l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation=tf.nn.relu)(l0)
        l2 = tf.keras.layers.Dropout(0.1)(l1)
        l3 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation=tf.nn.relu)(l2)
        l4 = get_varlayer(l3, 16, 1/training_batchsize)
        ys = [get_distlayer(l4, name=f"independent_normal_{t.label}") for t in targets]

    loss_fn = negative_loglikelihood
    model = tf.keras.Model(inputs=model_inputs, outputs=ys)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[r2])
    model.summary()
    return model


def model_aleotoric():
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(len(outputs), dtype=tf.float64), scale=1.0), reinterpreted_batch_ndims=1)
    # Define model instance.
    model = tfk.Sequential([
        tfk.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        tfk.layers.Dense(10, activation="relu", name="dense_1"),
        # tfp.layers.DenseFlipout(10, activation="relu", name="dense_1")
        tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(len(outputs)), activation=None,
                         name="distribution_weights"),
        # tfp.layers.DenseFlipout(10, activation="relu", name="dense_1")
        tfp.layers.MultivariateNormalTriL(
            len(outputs), activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1 / n_batches),
            name="output"
        )
    ], name="model")
    # Compile model.
    model.compile(optimizer="adam", loss=neg_log_likelihood)
    # Run training session.
    model.fit(data_train, epochs=n_epochs, validation_data=data_test, verbose=False)
    # Describe model.
    model.summary()