import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from DatasetCycle import DatasetCycle


class CycleGAN:
    def __init__(self, generator_g, generator_f, discriminator_x, discriminator_y, batch_size, epochs, filepath, checkpoint=None, train_steps=None,
                 val_steps=None, LAMBDA_g=.001, LAMBDA_f=.01, lr_bounds=None, lr_values=None):
        """
        Initializes the CycleGAN model with the given parameters.

        Args:
            generator_g: Generator model for translating X -> Y.
            generator_f: Generator model for translating Y -> X.
            discriminator_x: Discriminator model for domain X.
            discriminator_y: Discriminator model for domain Y.
            batch_size: Batch size for training.
            epochs: Number of epochs for training.
            filepath: Path to save checkpoints and logs.
            checkpoint: Optional checkpoint to restore the model.
            train_steps: Number of training steps per epoch.
            val_steps: Number of validation steps per epoch.
            LAMBDA_g: Weight for generator G loss.
            LAMBDA_f: Weight for generator F loss.
            lr_bounds: Learning rate boundaries for piecewise constant decay.
            lr_values: Learning rate values for piecewise constant decay.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.filepath = filepath
        self.LAMBDA_g = LAMBDA_g
        self.LAMBDA_f = LAMBDA_f

        self.loss_object = tf.keras.losses.MeanSquaredError() 

        if lr_bounds is None:
            lr_bounds = [10000, 30000]
        if lr_values is None:
            lr_values = [1*2e-4, .5*2e-4, .1*2e-4]

        lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_bounds, lr_values)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.5, beta_2=.99)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.5, beta_2=.99)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.5, beta_2=.99)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.5, beta_2=.99)

        self.generator_g = generator_g()
        self.generator_f = generator_f()
        self.generator_g.summary()
        self.discriminator_x = discriminator_x()
        self.discriminator_y = discriminator_y()
        self.discriminator_x.summary()

        dataset = DatasetCycle(self.batch_size)
        self.train_x_ds = dataset.train_x_ds
        self.train_y_ds = dataset.train_y_ds
        self.val_x_ds = dataset.val_x_ds
        self.val_y_ds = dataset.val_y_ds

        if train_steps is None:
            self.train_steps = dataset.train_steps
        else:
            self.train_steps = train_steps
        if val_steps is None:
            self.val_steps = dataset.val_steps
        else:
            self.val_steps = val_steps

        self.checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer)

        if checkpoint is not None:
            self.checkpoint.restore(checkpoint)

    def generator_loss(self, generated):
        """
        Calculates the generator loss.

        Args:
            generated: Discriminator output.

        Returns:
            Loss value.
        """
        return self.loss_object(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        Calculates the cycle consistency loss.

        Args:
            real_image: Real images.
            cycled_image: Cycled images.

        Returns:
            Loss value.
        """
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return loss

    def identity_loss(self, real_image, same_image):
        """
        Calculates the identity loss.

        Args:
            real_image: Real images.
            same_image: Generator output.

        Returns:
            Loss value.
        """
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Calculates the discriminator loss.

        Args:
            disc_real_output: Discriminator output for real images.
            disc_generated_output: Discriminator output for generated images.

        Returns:
            Tuple of total loss, real loss, and generated loss.
        """
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        disc_total_loss = (real_loss + generated_loss) / 2
        return disc_total_loss, real_loss, generated_loss

    def colorbar(self, img, fig, ax):
        """
        Adds a colorbar to the plot.

        Args:
            img: Image to add colorbar for.
            fig: Figure object.
            ax: Axes object.

        Returns:
            Colorbar object.
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cax=cax)
        return cbar

    @tf.function
    def train_step(self, real_x, real_y, step, summary_writer):
        """
        Performs a single training step.

        Args:
            real_x: Real images from domain X.
            real_y: Real images from domain Y.
            step: Current training step.
            summary_writer: Summary writer for TensorBoard.
        """
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            cycle_x_loss = self.calc_cycle_loss(real_x, cycled_x)
            cycle_y_loss = self.calc_cycle_loss(real_y, cycled_y)
            total_cycle_loss = cycle_x_loss + cycle_y_loss

            identity_loss_g = self.identity_loss(real_y, same_y)
            identity_loss_f = self.identity_loss(real_x, same_x)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = self.LAMBDA_g*gen_g_loss + total_cycle_loss + identity_loss_g/2
            total_gen_f_loss = self.LAMBDA_f*gen_f_loss + total_cycle_loss + identity_loss_f/2

            disc_x_loss, disc_x_real_loss, disc_x_gen_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss, disc_y_real_loss, disc_y_gen_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

        losses = {'gen_g_loss': gen_g_loss,
                  'gen_f_loss': gen_f_loss,
                  'cycle_x_loss': cycle_x_loss,
                  'cycle_y_loss': cycle_y_loss,
                  'total_cycle_loss': total_cycle_loss,
                  'identity_loss_g': identity_loss_g,
                  'identity_loss_f': identity_loss_f,
                  'total_gen_g_loss': total_gen_g_loss,
                  'total_gen_f_loss': total_gen_f_loss,
                  'disc_x_total_loss': disc_x_loss,
                  'disc_x_real_loss': disc_x_real_loss,
                  'disc_x_gen_loss': disc_x_gen_loss,
                  'disc_y__total_loss': disc_y_loss,
                  'disc_y_real_loss': disc_y_real_loss,
                  'disc_y_gen_loss': disc_y_gen_loss}

        with summary_writer.as_default():
            for key, value in losses.items():
                tf.summary.scalar(key, value, step=step)

    def val_test(self, input_x, input_y, plot=True, number=None, epoch=None):
        """
        Performs validation testing.

        Args:
            input_x: Input images from domain X.
            input_y: Input images from domain Y.
            plot: Whether to plot the results.
            number: Optional number for saving the plot.
            epoch: Optional epoch number for saving the plot.

        Returns:
            Dictionary of average generator G loss.
        """
        fake_y = self.generator_g(input_x, training=False)
        disc_fake_y = self.discriminator_y(fake_y, training=False)
        cycle_x = self.generator_f(fake_y, training=False)

        fake_x = self.generator_f(input_y, training=False)
        disc_fake_x = self.discriminator_x(fake_x, training=False)
        cycle_y = self.generator_g(fake_x, training=False)

        gen_g_loss = self.generator_loss(disc_fake_y)

        if plot:
            title = ['Input Image', 'Generated Image', 'Cycle Image']

            fig, axes = plt.subplots(self.batch_size*2, 4, figsize=(5*4, 5*2 * self.batch_size), dpi=150, squeeze=False)
            for img in range(self.batch_size):
                display_list = [[input_x[img], fake_y[img], cycle_x[img]], [input_y[img], fake_x[img], cycle_y[img]]]
                for i in range(3):
                    ax = axes[img*2, i]
                    ax.set_title(title[i])
                    # Getting the pixel values in the [0, 1] range to plot.
                    cb = ax.imshow(display_list[0][i] * 0.5 + 0.5, cmap='gray', extent=(-1.9, 1.915, 4.0180769, 0), vmin=0,
                                   vmax=1)
                    self.colorbar(cb, fig, ax)

                    ax = axes[img*2+1, i]
                    ax.set_title(title[i])
                    # Getting the pixel values in the [0, 1] range to plot.
                    cb = ax.imshow(display_list[1][i] * 0.5 + 0.5, cmap='gray', extent=(-1.9, 1.915, 4.0180769, 0), vmin=0,
                                   vmax=1)
                    self.colorbar(cb, fig, ax)

                ax = axes[img*2, 3]
                ax.set_title('Discriminator')
                cb = ax.imshow(disc_fake_y[img, ..., 0], cmap='RdBu_r', vmin=-10, vmax=10,
                               extent=(-1.9, 1.915, 4.0180769, 0))
                self.colorbar(cb, fig, ax)

                ax = axes[img*2+1, 3]
                ax.set_title('Discriminator')
                cb = ax.imshow(disc_fake_x[img, ..., 0], cmap='RdBu_r', vmin=-10, vmax=10,
                               extent=(-1.9, 1.915, 4.0180769, 0))
                self.colorbar(cb, fig, ax)
            fig.tight_layout()

            if number is not None and epoch is not None:
                plt.savefig(self.filepath + f'/val_img_epoch-{epoch:04d}-{number:02d}')
            plt.close(fig)
        metrics = {'avg gen_g_loss': gen_g_loss}
        return metrics

    def validate(self, epoch, summary_writer):
        """
        Validates the model.

        Args:
            epoch: Current epoch number.
            summary_writer: Summary writer for TensorBoard.
        """
        total_metrics = np.zeros(1)
        val_x_iter = iter(self.val_x_ds)
        val_y_iter = iter(self.val_y_ds)

        for val_step in range(self.val_steps):
            input_x = next(val_x_iter)
            input_y = next(val_y_iter)

            if self.batch_size == 1:
                if val_step < 5:
                    plot = True
                else:
                    plot = False
            else:
                if val_step < np.ceil(5/self.batch_size):
                    plot = True
                else:
                    plot = False
            metrics = self.val_test(input_x, input_y, plot, val_step, epoch)

            metric_values = list(metrics.values())
            for i in range(len(metrics)):
                total_metrics[i] += metric_values[i]

        metric_keys = list(metrics.keys())

        with summary_writer.as_default():
            for i in range(len(metrics)):
                tf.summary.scalar(metric_keys[i], total_metrics[i] / self.val_steps, step=epoch)

    def test(self, input_image):
        """
        Tests the model on a single input image.

        Args:
            input_image: Input image to test.

        Returns:
            Prints the metrics averaged over the batch size.
        """
        metrics = self.val_test(input_image)
        print(f'Metrics averaged over {self.batch_size} images')
        for key, value in metrics.items():
            print(f'{key}: {value}')

    def fit(self, restore_filepath=None):
        """
        Trains the model.

        Args:
            restore_filepath: Optional filepath to restore the latest checkpoint.
        """
        checkpoint_dir = self.filepath + '/checkpoints/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        if restore_filepath is not None:
            self.checkpoint.restore(tf.train.latest_checkpoint(restore_filepath))
            print('********** Restored Latest Checkpoint **********')

        log_dir = self.filepath + '/logs/'
        summary_writer = tf.summary.create_file_writer(log_dir)
        print(f'Tensorboard: tensorboard dev upload --logdir {log_dir}')

        start = time.time()
        step_count = 1
        train_x_iter = iter(self.train_x_ds)
        train_y_iter = iter(self.train_y_ds)

        for epoch in range(1, self.epochs + 1):
            print(f'\n********** Starting epoch {epoch}, Training... **********')
            for step in range(1, self.train_steps + 1):
                x = next(train_x_iter)
                y = next(train_y_iter)

                step_count_tensor = tf.convert_to_tensor(step_count, dtype=tf.int64)
                self.train_step(x, y, step_count_tensor, summary_writer)

                if step % 10 == 0:
                    print('.', end='', flush=True)
                if step % 1000 == 0:
                    print(
                        f' Time taken for 1000 steps: {time.time() - start:.2f} sec. Trained on {step * self.batch_size} images')
                    start = time.time()

                step_count += 1

            print(f'\n********** Completed epoch {epoch}, Saving and Validating... **********')
            self.checkpoint.save(file_prefix=checkpoint_prefix)
            self.validate(epoch, summary_writer)

        print('Successfully Trained!!!')
