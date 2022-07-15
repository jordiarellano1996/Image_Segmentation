import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class MyImageGenerator:
    def __init__(self, batch_size, img_size, custom_preprocessing_function=None, rescale=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.datagen = ImageDataGenerator(rescale=rescale, preprocessing_function=custom_preprocessing_function)

    def get_generator(self, df, x_col, y_col, color_mode, seed, shuffle, class_mode):
        gen = self.datagen.flow_from_dataframe(df, x_col=x_col, y_col=y_col,
                                               color_mode=color_mode,
                                               batch_size=self.batch_size, class_mode=class_mode,
                                               target_size=self.img_size, seed=seed, shuffle=shuffle)
        return gen


class MyCustomImageGenerator(MyImageGenerator):
    def __init__(self, batch_size, img_size):
        super().__init__(batch_size, img_size, custom_preprocessing_function=self._custom_preprocessing)

    def _custom_preprocessing(self, image_in):
        image = image_in.astype('float32')
        # Scale to [0, 255]
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        rescale = image * (1. / 255)
        return rescale

    def get_img_mask_generator(self, train_df, train_col, mask_col, seed, shuffle):
        """
        SEED--> So that image_generator and mask_generator will rotate and shuffle equivalently.
        In this situation the segmentation output is 'rgb' three channels, if the segmentation where > 3 channel I think
        this method would not work.
        """
        train_generator = self.get_generator(train_df, train_col, train_col, "grayscale", seed, shuffle, None)
        mask_generator = self.get_generator(train_df, mask_col, mask_col, "rgb", seed, shuffle, None)
        return train_generator, mask_generator

