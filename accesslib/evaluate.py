import os
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from sklearn.model_selection import StratifiedGroupKFold

from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef
from accesslib.model.image_generator import MyCustomImageGenerator
from accesslib import CFG
from accesslib.plot_factory.images import show_img


def set_seed(seed=25):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    np.random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


if __name__ == "__main__":
    cfg = CFG()
    set_seed(cfg.seed)

    # 🚀 Get data
    df = pd.read_csv(os.path.join(cfg.base_path, "train_precomputed.csv"))
    if cfg.debug:
        cases = np.random.choice(df.case.unique(), size=(cfg.debug_cases,))
    else:
        cases = df.case.unique()

    df = df[df.case.isin(cases)]
    df['segmentation'] = df.segmentation.fillna('')
    df = df.groupby(['id']).first()
    df["path_mask"].replace("/np/", "/png/", regex=True, inplace=True)
    df["path_mask"].replace(".npy", ".png", regex=True, inplace=True)
    df["path_scaled"] = df["path"].replace("/data/", "/data_scaled/", regex=True)
    df["path"].replace("/home/titoare/Documents/ds/image_segmentation/input", cfg.base_path, regex=True, inplace=True)
    df["path_mask"].replace("/home/titoare/Documents/ds/image_segmentation/input", cfg.base_path, regex=True,
                            inplace=True)

    with open(os.path.join(cfg.base_path, "cv_index"), "rb") as fp:
        cross_index = pickle.load(fp)

    """ Using on fold """
    fold = 0
    train = df.iloc[cross_index[fold][0]]
    validation = df.iloc[cross_index[fold][1]]

    """ Getting generators"""
    generator_engine = MyCustomImageGenerator(cfg.batch_size, cfg.img_size)
    train_generator_img, train_generator_mask = generator_engine.get_img_mask_generator(train, "path", "path_mask",
                                                                                        cfg.seed, True)
    validation_generator_img, validation_generator_mask = generator_engine.get_img_mask_generator(validation, "path",
                                                                                                  "path_mask", cfg.seed,
                                                                                                  True)

    # 🚀 Load model
    """ if you use custom metrics to evaluate the model, when you loaded you must pass
     custom objects instead you will get an error."""
    custom_objects = custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        "jacard_coef": jacard_coef,
    }
    out_model = load_model("/home/titoare/Documents/ds/Image_Segmentation/train_results/debug_False_19_07_2022/complete_model",
                           custom_objects=custom_objects)

    # 🚀 Evaluate
    b = 4
    pos = 9
    img = train_generator_img[b][pos]
    mask = train_generator_mask[b][pos]
    show_img(np.uint8(img * 255), np.uint8(mask * 255))

    img = img.reshape((1, 320, 384, 1))
    pred_mask = out_model.predict(img)
    pred_mask = np.uint8(pred_mask > 0.75)
    show_img(np.uint8(img[0] * 255), np.uint8(pred_mask[0] * 255))
