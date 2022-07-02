import os
import sys
import numpy as np
import cv2
import pandas as pd
from accesslib.plot_factory.images import show_img
from sklearn.model_selection import StratifiedGroupKFold

"""0: Debug, 1: No Info, 2: No info/warnings, 3: No info/warnings/error logged."""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from accesslib.unet import create_model, create_callbacks
from accesslib.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef

sys.path.append("/mounted/")


class CFG:
    seed = 25
    debug = True  # set debug=False for Full Training
    debug_cases = 8
    if len(os.getcwd().split("/")) > 4:
        base_path = "/home/titoare/Documents/ds/image_segmentation/input"
    else:
        base_path = "/mounted/input"
    epochs_path = "/home/titoare/Documents/ds/image_segmentation/accesslib/models"
    exp_name = 'Baselinev2'
    comment = 'unet-efficientnet_b1-224x224-aug2-split2'
    model_name = 'Unet'
    backbone = 'efficientnet-b1'
    img_size = (320, 384)
    batch_size = 16
    epochs = 80
    n_fold = 5


def custom_preprocessing(image_in):
    image = image_in.astype('float32')
    # Scale to [0, 255]
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    rescale = image * (1. / 255)
    return rescale


def set_seed(seed=25):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


if __name__ == "__main__":
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.compat.v1.Session(config=config)
    cfg = CFG()
    set_seed(CFG.seed)

    # 🚀 Train data
    df = pd.read_csv(os.path.join(cfg.base_path, "train_precomputed.csv"))
    if cfg.debug:
        cases = np.random.choice(df.case.unique(), size=(cfg.debug_cases,))
    else:
        cases = df.case.unique()

    df = df[df.case.isin(cases)]
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len)  # length of each rle mask
    df = df.groupby(['id']).first()
    df['empty'] = (df['rle_len'] == 0)  # empty masks
    df["path_mask"].replace("/np/", "/png/", regex=True, inplace=True)
    df["path_mask"].replace(".npy", ".png", regex=True, inplace=True)
    df["path_scaled"] = df["path"].replace("/data/", "/data_scaled/", regex=True)
    df["path"].replace("/home/titoare/Documents/ds/image_segmentation/input", cfg.base_path, regex=True, inplace=True)
    df["path_mask"].replace("/home/titoare/Documents/ds/image_segmentation/input", cfg.base_path, regex=True,
                            inplace=True)

    # 🚀 Cross validation
    """
    - Stratified K-Folds iterator variant with non-overlapping groups.
    - This cross-validation object is a variation of StratifiedKFold attempts to return stratified
      folds with non-overlapping groups. The folds are made by preserving the percentage of samples for each class.
    - The same group will not appear in two different folds (the number of distinct groups has to be at least equal to
      the number of folds).
    - The difference between GroupKFold and StratifiedGroupKFold is that the former attempts to create balanced
      folds such that the number of distinct groups is approximately the same in each fold, whereas StratifiedGroupKFold
      attempts to create folds which preserve the percentage of samples for each class as much as possible given the
      constraint of non-overlapping groups between splits.
    """
    skf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    cross_index = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
        cross_index.append((train_idx, val_idx))

    del fold, train_idx, val_idx, skf

    """ Using on fold """
    fold = 0
    train = df.iloc[cross_index[fold][0]]
    validation = df.iloc[cross_index[fold][1]]

    """
    SEED--> So that image_generator and mask_generator will rotate and shuffle equivalently.
    In this situation the segmentation output is 'rgb' three channels, if the segmentation where > 3 channel I think
    this method would not work.
    """
    datagen = ImageDataGenerator(rescale=None, preprocessing_function=custom_preprocessing)
    train_generator_img = datagen.flow_from_dataframe(train, x_col='path', y_col="path_mask",
                                                      color_mode="grayscale",
                                                      batch_size=cfg.batch_size, class_mode=None,
                                                      target_size=cfg.img_size, seed=cfg.seed, shuffle=True)
    train_generator_mask = datagen.flow_from_dataframe(train, x_col='path_mask', y_col="path", color_mode="rgb",
                                                       batch_size=cfg.batch_size, class_mode=None,
                                                       target_size=cfg.img_size, seed=cfg.seed, shuffle=True)

    train_generator = zip(train_generator_img, train_generator_mask)

    datagen = ImageDataGenerator(rescale=None, preprocessing_function=custom_preprocessing)
    validation_generator_img = datagen.flow_from_dataframe(validation, x_col='path', y_col="path_mask",
                                                           color_mode="grayscale",
                                                           batch_size=cfg.batch_size, class_mode=None,
                                                           target_size=cfg.img_size, seed=cfg.seed, shuffle=True)
    validation_generator_mask = datagen.flow_from_dataframe(validation, x_col='path_mask', y_col="path",
                                                            color_mode="rgb",
                                                            batch_size=cfg.batch_size, class_mode=None,
                                                            target_size=cfg.img_size, seed=cfg.seed, shuffle=True)

    validation_generator = zip(validation_generator_img, validation_generator_mask)

    # 🚀 Train
    model = create_model(n_classes=3, img_height=cfg.img_size[0], img_width=cfg.img_size[1], img_channels=1)
    callbacks = create_callbacks(cfg.epochs_path)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train) // cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=len(validation) // cfg.batch_size,

    )

    model.save(os.path.join(cfg.base_path, 'complete_model3'))
    from keras.models import load_model

    # if you use custom metrics to evaluate the model, when you loaded you must pass custom objects instead you will
    # get an error.
    custom_objects = custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        "jacard_coef": jacard_coef,
    }
    out_model = load_model(os.path.join(cfg.base_path, 'complete_model3'), custom_objects=custom_objects)
    # pred_mask = out_model.predict(validation_generator_img)
    # pred_mask = np.uint8(pred_mask > 0.65)
    #
    # b = 14
    # pos = 2
    # show_img(np.uint8(validation_generator_img[b][pos] * 255))
    # img = validation_generator_img[b][pos]
    # img = img.reshape((1, 320, 384, 1))
    # pred_mask = model.predict(img)
    # pred_mask = np.uint8(pred_mask > 0.65)
    # show_img(np.uint8(img[0] * 255), np.uint8(pred_mask[0] * 255))