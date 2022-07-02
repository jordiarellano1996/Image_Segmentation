import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_image(path):
    """
    Reads and converts the image.
    path: the full complete path to the .png file
    """

    # Read image in a corresponding manner
    # convert int16 -> float32
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    # Scale to [0, 255]
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image.astype(np.uint8)

    return image


def show_simple_images(sample_paths, mask=None):
    """Displays simple images (without mask)."""

    # Get additional info from the path
    case_name = [info.split("_")[-7].split("/")[-1] for info in sample_paths]
    day_name = [info.split("_")[-6].split("/")[0] for info in sample_paths]
    slice_name = [info.split("_")[-5] for info in sample_paths]

    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(23, 8))
    axs = axs.flatten()

    for k, path in enumerate(sample_paths):
        title = f"{k + 1}. {case_name[k]} - {day_name[k]} - {slice_name[k]}"
        axs[k].set_title(title, fontsize=14, weight='bold')

        img = read_image(path)
        axs[k].imshow(img, alpha=1)
        axs[k].axis("off")

        if mask is not None:
            mask_largeB = mask[k][:, :, 0]
            mask_smallB = mask[k][:, :, 1]
            mask_stomach = mask[k][:, :, 2]
            axs[k].imshow(mask_largeB, alpha=0.8 * (mask_largeB > 0), cmap="bwr")
            axs[k].imshow(mask_smallB, alpha=0.8 * (mask_smallB > 0), cmap="PiYG")
            axs[k].imshow(mask_stomach, alpha=0.8 * (mask_stomach > 0), cmap="RdBu")

            handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                       [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(handles, labels)

    plt.tight_layout()
    plt.show()


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
