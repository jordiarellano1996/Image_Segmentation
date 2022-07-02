import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_image_path(base_path, df):
    """
    Gets the case, day, slice_no and path of the dataset (either train or test).
    base_path: path to train image folder
    return :: modified dataframe
    """

    # Create case, day and slice columns
    df["case"] = df["id"].apply(lambda x: x.split("_")[0])
    df["day"] = df["id"].apply(lambda x: x.split("_")[1])
    df["slice_no"] = df["id"].apply(lambda x: x.split("_")[-1])

    df["path"] = 0

    n = len(df)

    # Loop through entire dataset
    for k in tqdm(range(n)):
        data = df.iloc[k, :]

        # In case coordinates for healthy tissue are present
        # if not pd.isnull(df.iloc[k, 2]):
        case = data.case
        day = data.day
        slice_no = data.slice_no
        # Change value to the correct one
        df.loc[k, "path"] = glob.glob(f"{base_path}/{case}/{case}_{day}/scans/slice_{slice_no}*")[0]

    return df


# Functions to get image width and height
def _get_img_size(x, flag):
    if x != 0:
        split = x.split("_")
        width = split[-4]
        height = split[-3]

        if flag == "width":
            return int(width)
        elif flag == "height":
            return int(height)

    return 0


def _get_pixel_size(x, flag):
    if x != 0:
        split = x.split("_")
        width = split[-2]
        height = ".".join(split[-1].split(".")[:-1])

        if flag == "width":
            return float(width)
        elif flag == "height":
            return float(height)

    return 0


def get_width_height(df):
    # Retrieve image width and height
    df["image_width"] = df["path"].apply(lambda x: _get_img_size(x, "width"))
    df["image_height"] = df["path"].apply(lambda x: _get_img_size(x, "height"))

    df["pixel_width"] = df["path"].apply(lambda x: _get_pixel_size(x, "width"))
    df["pixel_height"] = df["path"].apply(lambda x: _get_pixel_size(x, "height"))
    return df


# Segmentation and mask
def mask_from_segmentation(segmentation, shape):
    """
    Returns the mask corresponding to the inputed segmentation.
    segmentation: a list of start points and lengths in this order
    max_shape: the shape to be taken by the mask
    return:: a 2D mask
    """

    # Get a list of numbers from the initial segmentation
    segm = np.asarray(segmentation.split(), dtype=int)

    # Get start point and length between points
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Compute the location of each endpoint
    end_point = start_point + length_point

    # Create an empty list mask the size of the original image
    # take onl
    case_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Change pixels from 0 to 1 that are within the segmentation
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = 1

    case_mask = case_mask.reshape((shape[0], shape[1]))

    return case_mask


def get_id_mask(df, ID):
    """
    Returns a mask for each case ID. If no segmentation was found, the mask will be empty
    - meaning formed by only 0
    ID: the case ID from the train.csv file
    verbose: True if we want any prints
    return: segmentation mask
    """

    # ~~~ Get the data ~~~
    # Get the portion of dataframe where we have ONLY the speciffied ID
    ID_data = df[df["id"] == ID].reset_index(drop=True)

    # Split the dataframe into 3 series of observations
    # each for one speciffic class - "large_bowel", "small_bowel", "stomach"
    observations = [ID_data.loc[k, :] for k in range(3)]
    i = 0
    for obs in observations:
        if type(obs.segmentation) == float:
            i += 1
    no_segmentation = i == len(observations)

    # ~~~ Create the mask ~~~
    # Get the maximum height out of all observations
    # if max == 0 then no class has a segmentation
    # otherwise we keep the length of the mask
    max_height = np.max([obs.image_height for obs in observations])
    max_width = np.max([obs.image_width for obs in observations])

    # Get shape of the image
    # 3 channels of color/classes
    shape = (max_height, max_width, 3)

    # Create an empty mask with the shape of the image
    mask = np.zeros(shape, dtype=np.uint8)

    # If there is at least 1 segmentation found in the group of 3 classes
    if not no_segmentation:
        for k, location in enumerate(["large_bowel", "small_bowel", "stomach"]):
            observation = observations[k]
            segmentation = observation.segmentation

            # If a segmentation is found
            # Append a new channel to the mask
            if pd.isnull(segmentation) == False:
                mask[..., k] = mask_from_segmentation(segmentation, shape)
    return mask


# def save_image_on_disk(path, path_scaled):
#     dir_path = "/" + os.path.join(*path_scaled.split("/")[:-1])
#     try:
#         os.makedirs(dir_path)
#     except FileExistsError:
#         print(f"File exist")
#     cv2.imwrite(path_scaled, read_image(path))
