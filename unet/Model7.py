import tensorflow as tf
# import tensorflow_addons as tfa


class ModelBase:
    def __init__(self, kernel_size=3, gen_filters=None, disc_filters=None, deep_ds=True, deep_us=True, pair_disc=False,
                 gen_norm=None, disc_norm=None, dropout=None):
        """
        Initializes the ModelBase class with the given parameters.

        Args:
            kernel_size: Size of the convolutional kernels.
            gen_filters: List of filter sizes for the generator.
            disc_filters: List of filter sizes for the discriminator.
            deep_ds: Whether to use deep downsampling.
            deep_us: Whether to use deep upsampling.
            pair_disc: Whether to use paired discriminator.
            gen_norm: Normalization type for the generator.
            disc_norm: Normalization type for the discriminator.
            dropout: Dropout rate.
        """
        if gen_filters is None:
            gen_filters = [16, 16, 16, 16, 16]
        if disc_filters is None:
            disc_filters = [32, 64, 128, 256, 512]

        self.TARGET_HEIGHT = 512
        self.TARGET_WIDTH = 512

        self.ks = kernel_size

        self.downf = gen_filters
        self.upf = list(reversed(gen_filters[:-1]))
        self.discf = disc_filters

        self.deep_ds = deep_ds
        self.deep_us = deep_us

        self.pair_disc = pair_disc

        self.gen_norm = gen_norm
        self.disc_norm = disc_norm
        self.dropout = dropout

    def norm_layer(self, norm):
        """
        Returns the normalization layer based on the given normalization type.

        Args:
            norm: Normalization type ('instance_norm' or 'batch_norm').

        Returns:
            Normalization layer.
        """
        if norm == 'instance_norm':
            return tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True,
                                                    beta_initializer='random_uniform',
                                                    gamma_initializer='random_uniform')
        if norm == 'batch_norm':
            return tf.keras.layers.BatchNormalization()


class Generator(ModelBase):
    def __init__(self, **kwargs):
        """
        Initializes the Generator class with the given parameters.
        """
        super().__init__(**kwargs)

    def __call__(self):
        """
        Builds the generator model.

        Returns:
            Generator model.
        """
        inputs = tf.keras.layers.Input(shape=[self.TARGET_HEIGHT, self.TARGET_WIDTH, 1])
        x = inputs

        skips = []
        for i, filter_num in enumerate(self.downf):
            x = tf.keras.layers.Conv2D(filter_num, self.ks, strides=1, padding='same')(x)
            x = tf.keras.layers.LeakyReLU()(x)
            if self.gen_norm is not None:
                x = self.norm_layer(self.gen_norm)(x)
            if self.dropout is not None:
                x = tf.keras.layers.Dropout(self.dropout)(x)

            if self.deep_ds:
                x = tf.keras.layers.Conv2D(filter_num, self.ks, strides=1, padding='same')(x)
                x = tf.keras.layers.LeakyReLU()(x)
                if self.gen_norm is not None:
                    x = self.norm_layer(self.gen_norm)(x)
                if self.dropout is not None:
                    x = tf.keras.layers.Dropout(self.dropout)(x)

            if i != len(self.downf) - 1:
                skips.append(x)
                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
                if self.gen_norm is not None:
                    x = self.norm_layer(self.gen_norm)(x)
                if self.dropout is not None:
                    x = tf.keras.layers.Dropout(self.dropout)(x)


        skips.reverse()

        for i, filter_num in enumerate(self.upf):
            x = tf.keras.layers.Conv2DTranspose(filter_num, self.ks, strides=2, padding='same')(x)
            x = tf.keras.layers.Concatenate()([x, skips[i]])

            x = tf.keras.layers.Conv2D(filter_num, self.ks, strides=1, padding='same')(x)
            x = tf.keras.layers.ReLU()(x)
            if self.gen_norm is not None:
                x = self.norm_layer(self.gen_norm)(x)
            if self.dropout is not None:
                x = tf.keras.layers.Dropout(self.dropout)(x)

            if self.deep_us:
                x = tf.keras.layers.Conv2D(filter_num, self.ks, strides=1, padding='same')(x)
                x = tf.keras.layers.ReLU()(x)
                if self.gen_norm is not None:
                    x = self.norm_layer(self.gen_norm)(x)
                if self.dropout is not None:
                    x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same')(x)  # MimickNet does not use tanh

        return tf.keras.Model(inputs=inputs, outputs=x)


class Discriminator(ModelBase):
    def __init__(self, **kwargs):
        """
        Initializes the Discriminator class with the given parameters.
        """
        super().__init__(**kwargs)

    def __call__(self):
        """
        Builds the discriminator model.

        Returns:
            Discriminator model.
        """
        if self.pair_disc:
            inp = tf.keras.layers.Input(shape=[self.TARGET_HEIGHT, self.TARGET_WIDTH, 1], name='input_image')
            tar = tf.keras.layers.Input(shape=[self.TARGET_HEIGHT, self.TARGET_WIDTH, 1], name='target_image')
            inputs = [inp, tar]
            x = tf.keras.layers.concatenate(inputs)  # (batch_size, 512, 512, channels*2)
        else:
            inputs = tf.keras.layers.Input(shape=[self.TARGET_HEIGHT, self.TARGET_WIDTH, 1])
            x = inputs

        for i, filter_num in enumerate(self.discf):
            if self.disc_norm == 'spectral':
                x = tfa.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(filter_num, self.ks, strides=1, padding='same'))(x)
                x = tf.keras.layers.ReLU()(x)  # MimickNet uses ReLU instead of leakyReLU

                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)


            else:
                x = tf.keras.layers.Conv2D(filter_num, self.ks, strides=1, padding='same')(x)
                x = tf.keras.layers.ReLU()(x)
                if self.disc_norm is not None:
                    x = self.norm_layer(self.disc_norm)(x)

                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Conv2D(1, self.ks, strides=1, padding='same')(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
