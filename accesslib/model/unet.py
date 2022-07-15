"""
Standard Unet
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPooling2D, Input, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef
import wandb
from wandb.keras import WandbCallback


def create_model(n_classes=3, img_height=320, img_width=320, img_channels=1, verbose=True):
    """
    :return:
    """
    """
    UpSampling2D is just a simple scaling up of the image by using nearest neighbour or bilinear upsampling,
    so nothing smart. Advantage is it's cheap.
    Conv2DTranspose is a convolution operation whose kernel is learnt (just like normal conv2d operation) while
    training your model. Using Conv2DTranspose will also upsample its input but the key difference is the
    model should learn what is the best upsampling for the job.
    """
    # Input layer
    input_layer = Input((img_height, img_width, img_channels))

    # Encoding path
    c1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding='same')(input_layer)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding='same')(c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same')(c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    c3 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same')(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same')(c3)
    p3 = MaxPooling2D((2, 2), strides=(2, 2))(c3)

    c4 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same')(c4)
    p4 = MaxPooling2D((2, 2), strides=(2, 2))(c4)

    # Lower layer
    c5 = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same')(c6)

    u7 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same')(c7)

    u8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same')(c8)

    u9 = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding='same')(c9)

    # output = Conv1D(filters=3, kernel_size=1, strides=1, activation="sigmoid")(c9)
    output = Conv2D(filters=n_classes, kernel_size=(1, 1), activation="sigmoid", padding='same')(
        c9)  # hot encoded output

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # 228 - Semantic segmentation of aerial (satellite) imagery using U-net
    # If teh class are not balanced, you can compute class weights and applied to DiceLoss
    # Check focal loss also
    model.compile(optimizer=opt,
                  loss=bce_dice_loss, metrics=[dice_coef, iou_coef, jacard_coef],
                  )

    if verbose:
        print(model.summary())

    return model


def create_callbacks(path, wandb_flag=False, wandb_test_name="NoTestName"):
    """"""
    filename = "/RNN_Final-{epoch:02d}-{loss:.5f}"
    checkpoint = ModelCheckpoint("{}{}.model".format(path, filename,
                                                     monitor='loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max'))  # saves only the best ones.
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    if wandb_flag:
        wandb_config = {'competition': 'AWMadison', 'GPU_name': 'RTX6000 GPU x1'}
        wandb.login(relogin=True, key="52fb822f6a358eedc0a801169d4b00b63ffa125f")
        run = wandb.init(project="ImageSegmentation", entity="jordiarellano1996",
                         name=wandb_test_name, config=wandb_config)
        return [early_stop, checkpoint, WandbCallback()]
    else:
        return [early_stop, checkpoint]
