import tensorflow as tf
import pandas as pd
import numpy as np


class DatasetCycle:
    def __init__(self, batch_size=1):
        """
        Initializes the DatasetCycle class with the given batch size.

        Args:
            batch_size: Batch size for the dataset.
        """
        self.TARGET_HEIGHT = 512
        self.TARGET_WIDTH = 512
        self.CROP_HEIGHT = 448  # 512*7/8
        self.CROP_WIDTH = 448

        self.BATCH_SIZE = batch_size

        # Load training, validation, and test datasets for domain X
        train_x_df = pd.read_csv('./MimickNet/clean_PW_muscle_data/train_dataset.csv')
        train_x = train_x_df['filtered'].tolist()

        val_x_df = pd.read_csv('./MimickNet/clean_PW_muscle_data/val_dataset.csv')
        val_x = val_x_df['filtered'].tolist()

        test_x_df = pd.read_csv('./MimickNet/clean_PW_muscle_data/test_dataset.csv')
        test_x = test_x_df['filtered'].tolist()

        # Load training, validation, and test datasets for domain Y
        train_y_df = pd.read_csv('./MimickNet/online_muscle_datasets_grayscale/train_dataset.csv')
        train_y = train_y_df['filename'].tolist()

        val_y_df = pd.read_csv('./MimickNet/online_muscle_datasets_grayscale/val_dataset.csv')
        val_y = val_y_df['filename'].tolist()

        test_y_df = pd.read_csv('./MimickNet/online_muscle_datasets_grayscale/test_dataset.csv')
        test_y = test_y_df['filename'].tolist()

        # Create TensorFlow datasets
        self.train_x_ds, train_x_size = self.make_train_ds(train_x)
        self.val_x_ds, val_x_size = self.make_valtest_ds(val_x)
        self.test_x_ds, test_x_size = self.make_valtest_ds(test_x)

        self.train_y_ds, train_y_size = self.make_train_ds(train_y)
        self.val_y_ds, val_y_size = self.make_valtest_ds(val_y)
        self.test_y_ds, test_y_size = self.make_valtest_ds(test_y)

        # Calculate the number of steps per epoch
        self.train_steps = np.minimum(train_x_size, train_y_size) // self.BATCH_SIZE
        self.val_steps = np.minimum(val_x_size, val_y_size) // self.BATCH_SIZE
        self.test_steps = np.minimum(test_x_size, test_y_size) // self.BATCH_SIZE

    def make_train_ds(self, df):
        """
        Creates a training dataset from the given dataframe.

        Args:
            df: Dataframe containing the file paths.

        Returns:
            A tuple of the training dataset and its size.
        """
        train = tf.data.Dataset.from_tensor_slices(df)
        train_size = train.cardinality().numpy()
        train = train.shuffle(train_size, reshuffle_each_iteration=True).repeat()
        train = train.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return train, train_size

    def make_valtest_ds(self, df):
        """
        Creates a validation or test dataset from the given dataframe.

        Args:
            df: Dataframe containing the file paths.

        Returns:
            A tuple of the validation/test dataset and its size.
        """
        valtest = tf.data.Dataset.from_tensor_slices(df)
        valtest_size = valtest.cardinality().numpy()
        valtest = valtest.map(self.load_image_val_test, num_parallel_calls=tf.data.AUTOTUNE)
        valtest = valtest.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return valtest, valtest_size

    def load_image_val_test(self, input_image):
        """
        Loads and preprocesses an image for validation or testing.

        Args:
            input_image: File path of the input image.

        Returns:
            Preprocessed image tensor.
        """
        input_image = tf.io.read_file(input_image)
        input_image = tf.io.decode_image(input_image, expand_animations=False)
        input_image = tf.cast(input_image, tf.float32)

        input_image = tf.image.resize(input_image, [self.TARGET_HEIGHT, self.TARGET_WIDTH])

        # Normalize the image to the range [-1, 1]
        input_image = input_image / 127.5 - 1

        return input_image

    def load_image_train(self, input_image):
        """
        Loads and preprocesses an image for training.

        Args:
            input_image: File path of the input image.

        Returns:
            Preprocessed image tensor.
        """
        input_image = tf.io.read_file(input_image)
        input_image = tf.io.decode_image(input_image, expand_animations=False)
        input_image = tf.cast(input_image, tf.float32)

        input_image = tf.image.resize(input_image, [self.TARGET_HEIGHT, self.TARGET_WIDTH])
        input_image = tf.image.random_crop(input_image, size=[self.CROP_HEIGHT, self.CROP_WIDTH, 1])
        input_image = tf.image.random_flip_left_right(input_image)
        input_image = tf.image.resize(input_image, [self.TARGET_HEIGHT, self.TARGET_WIDTH])

        # Normalize the image to the range [-1, 1]
        input_image = input_image / 127.5 - 1

        return input_image
