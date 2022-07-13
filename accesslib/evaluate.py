import os
import numpy as np
from keras.models import load_model
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef
from accesslib import CFG

if __name__ == "__main__":
    cfg = CFG()

    # if you use custom metrics to evaluate the model, when you loaded you must pass custom objects instead you will
    # get an error.
    custom_objects = custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        "jacard_coef": jacard_coef,
    }
    out_model = load_model(os.path.join(cfg.base_path, 'complete_model'), custom_objects=custom_objects)
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