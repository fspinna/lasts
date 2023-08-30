import pathlib
from joblib import dump, load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lasts.autoencoders.tools import repeat_block


class EncoderWrapper(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def predict(self, X):
        return self.encoder.predict(X)[2]


def save_model(
    model, input_shape, latent_dim, autoencoder_kwargs, path="weights", verbose=False
):
    path = pathlib.Path(path)
    model_kwargs = {
        "input_shape": input_shape,
        "latent_dim": latent_dim,
        "autoencoder_kwargs": autoencoder_kwargs,
    }
    model.save_weights(path.parents[0] / (path.name + ".h5"))
    dump(model_kwargs, path.parents[0] / (path.name + ".joblib"))
    return


def load_model(path="weights", verbose=False, wrap_encoder=True):
    path = pathlib.Path(path)
    model_kwargs = load(path.parents[0] / (path.name + ".joblib"))
    encoder, decoder, autoencoder = build_vae(
        model_kwargs.get("input_shape"),
        model_kwargs.get("latent_dim"),
        model_kwargs.get("autoencoder_kwargs"),
        verbose=verbose,
    )
    autoencoder.load_weights(path.parents[0] / (path.name + ".h5"))
    if wrap_encoder:
        encoder = EncoderWrapper(encoder)
    return encoder, decoder, autoencoder


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        return self.decoder(self.encoder(inputs)[2])

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = tf.keras.losses.MeanSquaredError(
                reduction="auto", name="mean_squared_error"
            )
            reconstruction_loss = mse(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        mse = tf.keras.losses.MeanSquaredError(
            reduction="auto", name="mean_squared_error"
        )
        reconstruction_loss = mse(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_encoder(input_shape, latent_dim, **kwargs):
    encoder_input = keras.layers.Input(shape=(input_shape))
    encoder_layers = repeat_block(
        encoder_input,
        "encoder",
        filters=kwargs.get("filters"),
        kernel_size=kwargs.get("kernel_size"),
        padding=kwargs.get("padding"),
        activation=kwargs.get("activation"),
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling"),
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual"),
    )
    encoder_layers = keras.layers.Conv1D(
        filters=input_shape[1],  # FIXME: or 1?
        kernel_size=1,  # FIXME: maybe different value?
        padding="same",
    )(encoder_layers)
    encoder_layers = keras.layers.Flatten()(encoder_layers)

    z_mean = layers.Dense(latent_dim, name="z_mean")(encoder_layers)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_layers)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="Encoder")
    return encoder


def build_decoder(encoder, latent_dim, **kwargs):
    decoder_input = keras.layers.Input(shape=(latent_dim,), name="z_sampling")
    decoder_layers = decoder_input
    decoder_layers = keras.layers.Dense(encoder.layers[-4].output_shape[1])(
        decoder_layers
    )
    decoder_layers = keras.layers.Reshape(encoder.layers[-5].output_shape[1:])(
        decoder_layers
    )

    decoder_layers = repeat_block(
        decoder_layers,
        "decoder",
        filters=kwargs.get("filters")[::-1],
        kernel_size=kwargs.get("kernel_size")[::-1],
        padding=kwargs.get("padding")[::-1],
        activation=kwargs.get("activation")[::-1],
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling")[::-1],
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual"),
    )
    decoder_output = keras.layers.Conv1D(
        filters=encoder.input_shape[2], kernel_size=1, padding="same"
    )(decoder_layers)
    decoder = keras.models.Model(decoder_input, decoder_output, name="Decoder")
    return decoder


def build_vae(
    input_shape,
    latent_dim,
    autoencoder_kwargs,
    verbose=True,
):
    encoder = build_encoder(input_shape, latent_dim, **autoencoder_kwargs)

    decoder = build_decoder(encoder, latent_dim, **autoencoder_kwargs)

    # model_input = keras.layers.Input(shape=input_shape)
    # eps = Input(tensor=K.random_normal(stddev=1, shape=(K.shape(model_input)[0], latent_dim)))
    # model_input = [model_input, eps]
    # model_output = decoder(encoder(model_input))
    # autoencoder = keras.models.Model(model_input, model_output, name="VAE")

    autoencoder = VAE(encoder, decoder)
    autoencoder.compile(
        optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        loss=None,
        metrics=["mse"],
    )
    autoencoder.build(input_shape[::-1])  # wants the n_timesteps on the -1 axis

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()

    return encoder, decoder, autoencoder


if __name__ == "__main__":
    from lasts.datasets.datasets import build_cbf
    from lasts.deprecated.blackboxes import blackbox_loader
    from lasts.deprecated.blackboxes import BlackboxWrapper
    from lasts.plots import plot_grouped_history
    from lasts.utils import plot_reconstruction_vae

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    ) = build_cbf(n_samples=600)

    input_shape = X_train.shape[1:]

    latent_dim = 2

    autoencoder_kwargs = {
        "filters": [4, 4, 4, 4],
        "kernel_size": [3, 3, 3, 3],
        "padding": ["same", "same", "same", "same"],
        "activation": ["relu", "relu", "relu", "relu"],
        "pooling": [1, 1, 1, 1],
        "n_layers": 4,
        "optimizer": "adam",
        "n_layers_residual": None,
        "batch_normalization": None,
    }

    blackbox = BlackboxWrapper(blackbox_loader("cbf_knn.joblib"), 2, 1)

    encoder, decoder, autoencoder = build_vae(
        input_shape, latent_dim, autoencoder_kwargs
    )

    hist = autoencoder.fit(X_exp_train, epochs=2000, validation_data=(X_exp_val,))

    plot_grouped_history(hist.history)

    encoder_w = EncoderWrapper(encoder)

    plot_reconstruction_vae(X_train[:5], encoder_w, decoder)
