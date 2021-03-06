import os


class CFG:
    seed = 25
    debug = False  # set debug=False for Full Training
    wandb_callback_flag = True
    wandb_test_name = "train_20_07_2022_unet_filter_division_2"
    debug_cases = 10
    if len(os.getcwd().split("/")) > 4:
        base_path = "/home/titoare/Documents/ds/Image_Segmentation/input"
    else:
        base_path = "/mounted/input"
    print(f"base_path: {base_path}")
    epochs_path = base_path + "/models"
    exp_name = 'Baselinev2'
    comment = 'unet-efficientnet_b1-224x224-aug2-split2'
    model_name = 'Unet'
    backbone = 'efficientnet-b1'
    img_size = (320, 384)
    batch_size = 64
    epochs = 100
    n_fold = 5


__version__ = "unknown"

" It is a list of strings defining what symbols in a module willbe exported when"
" from <module> import * is used on the module."

__all__ = [
    "CFG",
]
