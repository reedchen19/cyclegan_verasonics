import tensorflow as tf
import pandas as pd


class DatasetPW:
    def __init__(self, batch_size=1):
        """
        Initializes the DatasetPW class with the given batch size.

        Args:
            batch_size: Batch size for the dataset.
        """
        self.TARGET_HEIGHT = 512
        self.TARGET_WIDTH = 512
        self.CROP_HEIGHT = 448
        self.CROP_WIDTH = 448

        self.BATCH_SIZE = batch_size

        # Load training, validation, and test datasets
        train_df = pd.read_csv('./clean_PW_muscle_data/train_dataset.csv')
        train_df.head()
        train_images = train_df['original'].tolist()
        train_labels = train_df['filtered'].tolist()

        val_df = pd.read_csv('./clean_PW_muscle_data/val_dataset.csv')
        val_df.head()
        val_images = val_df['original'].tolist()
        val_labels = val_df['filtered'].tolist()

        test_df = pd.read_csv('./clean_PW_muscle_data/test_dataset.csv')
        test_df.head()
        test_images = test_df['original'].tolist()
        test_labels = test_df['filtered'].tolist()

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

        self.train_size = tf.data.experimental.cardinality(train_ds).numpy()
        self.val_size = tf.data.experimental.cardinality(val_ds).numpy()
        self.test_size = tf.data.experimental.cardinality(test_ds).numpy()

        self.train_steps = self.train_size // self.BATCH_SIZE
        self.val_steps = self.val_size // self.BATCH_SIZE
        self.test_steps = self.test_size // self.BATCH_SIZE

        print("train size: ", self.train_size)
        print("val size: ", self.val_size)
        print("test size: ", self.test_size)

        train_ds = train_ds.shuffle(self.train_size, reshuffle_each_iteration=True).repeat()

        train_ds = train_ds.map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_ds = train_ds.batch(self.BATCH_SIZE)

        val_ds = val_ds.map(self.load_image_val_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_ds = val_ds.batch(self.BATCH_SIZE)

        test_ds = test_ds.map(self.load_image_val_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_ds = test_ds.batch(self.BATCH_SIZE)

    def load(self, image_file, label_file):
        """
        Loads and decodes an image and its corresponding label.

        Args:
            image_file: File path of the input image.
            label_file: File path of the label image.

        Returns:
            Tuple of the input image and label image tensors.
        """
        image = tf.io.read_file(image_file)
        image = tf.io.decode_png(image)

        label = tf.io.read_file(label_file)
        label = tf.io.decode_png(label)

        # Convert both images to float32 tensors
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)

        return image, label

    def random_crop(self, input_image, real_image):
        """
        Randomly crops the input and label images.

        Args:
            input_image: Input image tensor.
            real_image: Label image tensor.

        Returns:
            Tuple of cropped input and label images.
        """
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, self.CROP_HEIGHT, self.CROP_WIDTH, 1])

        return cropped_image[0], cropped_image[1]

    def resize(self, input_image, real_image):
        """
        Resizes the input and label images to the target dimensions.

        Args:
            input_image: Input image tensor.
            real_image: Label image tensor.

        Returns:
            Tuple of resized input and label images.
        """
        input_image = tf.image.resize(input_image, [self.TARGET_HEIGHT, self.TARGET_WIDTH])
        real_image = tf.image.resize(real_image, [self.TARGET_HEIGHT, self.TARGET_WIDTH])
        return input_image, real_image

    def random_flip(self, input_image, real_image):
        """
        Randomly flips the input and label images horizontally.

        Args:
            input_image: Input image tensor.
            real_image: Label image tensor.

        Returns:
            Tuple of flipped input and label images.
        """
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
        return input_image, real_image

    def normalize(self, input_image, real_image):
        """
        Normalizes the input and label images to the range [-1, 1].

        Args:
            input_image: Input image tensor.
            real_image: Label image tensor.

        Returns:
            Tuple of normalized input and label images.
        """
        input_image = input_image / 127.5 - 1
        real_image = real_image / 127.5 - 1

        return input_image, real_image

    def augment(self, input_image, real_image):
        """
        Applies data augmentation to the input and label images.

        Args:
            input_image: Input image tensor.
            real_image: Label image tensor.

        Returns:
            Tuple of augmented input and label images.
        """
        input_image, real_image = self.random_crop(input_image, real_image)
        input_image, real_image = self.random_flip(input_image, real_image)
        return input_image, real_image

    def load_image_train(self, original, label):
        """
        Loads and preprocesses an image and its label for training.

        Args:
            original: File path of the input image.
            label: File path of the label image.

        Returns:
            Tuple of preprocessed input and label images.
        """
        input_image, real_image = self.load(original, label)
        input_image, real_image = self.resize(input_image, real_image)
        input_image, real_image = self.augment(input_image, real_image)
        input_image, real_image = self.resize(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_val_test(self, original, label):
        """
        Loads and preprocesses an image and its label for validation or testing.

        Args:
            original: File path of the input image.
            label: File path of the label image.

        Returns:
            Tuple of preprocessed input and label images.
        """
        input_image, real_image = self.load(original, label)
        input_image, real_image = self.resize(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image
