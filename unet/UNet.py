import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skimage.metrics import structural_similarity as ssim

from DatasetPW import DatasetPW


class UNet:
    def __init__(self, model, batch_size, epochs, filepath, checkpoint=None, train_steps=None, val_steps=None):
        """
        Initializes the UNet class with the given parameters.

        Args:
            model: The generator model.
            batch_size: Batch size for training.
            epochs: Number of epochs for training.
            filepath: Path to save checkpoints and logs.
            checkpoint: Optional checkpoint to restore the model.
            train_steps: Number of training steps per epoch.
            val_steps: Number of validation steps per epoch.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.filepath = filepath

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.generator = model()
        self.generator.summary()

        dataset = DatasetPW(self.batch_size)
        self.train_ds = dataset.train_ds
        if train_steps is None:
            self.train_steps = dataset.train_steps
        else:
            self.train_steps = train_steps
        self.val_ds = dataset.val_ds
        if val_steps is None:
            self.val_steps = dataset.val_steps
        else:
            self.val_steps = val_steps

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, generator=self.generator)

        if checkpoint is not None:
            self.checkpoint.restore(checkpoint)
            print('********** Restored Checkpoint **********')

    def generator_loss(self, gen_output, target):
        """
        Calculates the generator loss.

        Args:
            gen_output: Generated images.
            target: Target images.

        Returns:
            Loss value.
        """
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        return l1_loss

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

    def val_test(self, input_image, target, plot=True, number=None, epoch=None):
        """
        Performs validation testing.

        Args:
            input_image: Input images.
            target: Target images.
            plot: Whether to plot the results.
            number: Optional number for saving the plot.
            epoch: Optional epoch number for saving the plot.

        Returns:
            Dictionary of average generator L1 loss, SSIM before and after, and MSE before and after.
        """
        gen_output = self.generator(input_image, training=False)

        gen_l1_loss = self.generator_loss(gen_output, target)
        mse_before = tf.reduce_mean(tf.abs(input_image - target))
        mse_after = tf.reduce_mean(tf.abs(gen_output - target))

        ssim_before = 0
        ssim_after = 0
        for i in range(self.batch_size):
            ssim_before += ssim(np.squeeze(input_image[i]), np.squeeze(target[i]), win_size=11, data_range=2)
            ssim_after += ssim(np.squeeze(gen_output[i]), np.squeeze(target[i]), win_size=11, data_range=2)
        ssim_before = ssim_before / self.batch_size
        ssim_after = ssim_after / self.batch_size

        if plot:
            title = ['Input Image', 'Ground Truth', 'Predicted Image']

            fig, axes = plt.subplots(self.batch_size, 3, figsize=(15, 5 * self.batch_size), dpi=150, squeeze=False)
            for img in range(self.batch_size):
                display_list = [input_image[img], target[img], gen_output[img]]
                for i in range(3):
                    ax = axes[img, i]
                    ax.set_title(title[i])
                    # Getting the pixel values in the [0, 1] range to plot.
                    cb = ax.imshow(display_list[i] * 0.5 + 0.5, cmap='gray', extent=(-1.9, 1.915, 4.0180769, 0), vmin=0,
                                   vmax=1)
                    self.colorbar(cb, fig, ax)
            fig.tight_layout()

            if number is not None and epoch is not None:
                plt.savefig(self.filepath + f'/val_img_epoch-{epoch:04d}-{number:02d}')
            plt.close(fig)
        metrics = {'avg gen_l1_loss': gen_l1_loss, 'avg ssim_before': ssim_before, 'avg ssim_after': ssim_after,
                   'avg mse_before': mse_before, 'avg mse_after': mse_after}
        return metrics

    @tf.function
    def train_step(self, input_image, target, step, summary_writer):
        """
        Performs a single training step.

        Args:
            input_image: Input images.
            target: Target images.
            step: Current training step.
            summary_writer: Summary writer for TensorBoard.
        """
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(input_image, training=True)
            gen_l1_loss = self.generator_loss(gen_output, target)

        generator_gradients = gen_tape.gradient(gen_l1_loss,
                                                self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen l1_loss', gen_l1_loss, step=step)

    def validate(self, epoch, summary_writer):
        """
        Validates the model.

        Args:
            epoch: Current epoch number.
            summary_writer: Summary writer for TensorBoard.
        """
        total_metrics = np.zeros(5)
        val_ds_iter = iter(self.val_ds)

        for val_step in range(self.val_steps):
            input_image, target = next(val_ds_iter)

            if val_step < 5:
                plot = True
            else:
                plot = False
            metrics = self.val_test(input_image, target, plot, val_step, epoch)

            metric_values = list(metrics.values())
            for i in range(len(metrics)):
                total_metrics[i] += metric_values[i]

        metric_keys = list(metrics.keys())

        with summary_writer.as_default():
            for i in range(len(metrics)):
                tf.summary.scalar(metric_keys[i], total_metrics[i] / self.val_steps, step=epoch)

    def test(self, input_image, target):
        """
        Tests the model on a single input image.

        Args:
            input_image: Input image to test.
            target: Target image.

        Returns:
            Prints the metrics averaged over the batch size.
        """
        metrics = self.val_test(input_image, target)
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
        train_ds_iter = iter(self.train_ds)

        for epoch in range(1, self.epochs + 1):
            print(f'\n********** Starting epoch {epoch}, Training... **********')
            for step in range(1, self.train_steps + 1):
                input_image, target = next(train_ds_iter)
                step_count_tensor = tf.convert_to_tensor(step_count, dtype=tf.int64)
                self.train_step(input_image, target, step_count_tensor, summary_writer)

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
