import os
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from accesslib.plot_factory.seaborn import show_values_on_bars
from accesslib.plot_factory.wandb import create_wandb_plot
from accesslib.plot_factory.images import show_simple_images, read_image, show_img
from accesslib import CFG


class CLR:
    S = '\033[1m' + '\033[92m'
    E = '\033[0m'


def save_dataset_artifact(run_name, artifact_name, path, cfg_wandb_config):
    '''Saves dataset to W&B Artifactory.
    run_name: name of the experiment
    artifact_name: under what name should the dataset be stored
    path: path to the dataset'''

    run = wandb.init(project='AWMadison',
                     name=run_name,
                     config=cfg_wandb_config)
    artifact = wandb.Artifact(name=artifact_name,
                              type='dataset')
    artifact.add_file(path)

    wandb.log_artifact(artifact)
    wandb.finish()
    print("Artifact has been saved successfully.")


if __name__ == "__main__":
    cfg = CFG()
    clr = CLR()
    wandb.login(key="52fb822f6a358eedc0a801169d4b00b63ffa125f")

    """ 🤫 Experiment 1: Explore train csv data. """
    # 🐝 Log Cover Photo
    if cfg.wandb_log:
        run = wandb.init(project='AWMadison', name='CoverPhoto', config=cfg.wandb_config)
        cover = plt.imread("/home/titoare/Documents/ds/image_segmentation/input/cover.png")
        wandb.log({"example": wandb.Image(cover)})
        wandb.finish()

    # 🚀 Train data shape info
    train = pd.read_csv(os.path.join(cfg.base_path, "train_precomputed.csv"))

    print(clr.S + "shape:" + clr.E, train.shape)
    print(clr.S + "Unique ID cases:" + clr.E, train["id"].nunique())
    print(clr.S + "Missing Values Column:" + clr.E, train.isna().sum().index[2])
    print("\t", clr.S + "with a total missing rows of:" + clr.E, train.isna().sum().values[2])
    print("\t", clr.S + "% of missing rows:" + clr.E,
          len(train[train["segmentation"].isna() == False]), "\n")

    print(clr.S + "Sample of train.csv:" + clr.E)
    print(train.sample(5, random_state=26))
    # 🐝
    if cfg.wandb_log:
        run = wandb.init(project='AWMadison', name='data_explore', config=cfg.wandb_config)
        wandb.log({"train_len": train.shape[0],
                   "train_cols": train.shape[1],
                   "segmentation_no": len(train[train["segmentation"].isna() == False]),
                   "segmentation_perc": round(
                       (len(train[train["segmentation"].isna() == False]) / train.shape[0]) * 100, 1)})

    # 🚀 ID interpretability:
    """Make the columns unique as they repeat sometimes due to multiple "class" values"""
    data = train.groupby("id")[["case", "day", "slice_no"]].first().reset_index()
    data["slice_no"] = data["slice_no"].astype(str)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 35))
    titles = ["Case", "Day", "Slice No."]

    sns.barplot(data=data["case"].value_counts().reset_index(),
                y="index", x="case", ax=ax1,
                palette="YlOrBr_r")

    sns.barplot(data=data["day"].value_counts().reset_index(),
                y="index", x="day", ax=ax2,
                palette="YlGn_r")

    sns.barplot(data=data["slice_no"].value_counts().reset_index(),
                y="index", x="slice_no", ax=ax3,
                palette="Greens_r")

    for ax, t in zip([ax1, ax2, ax3], titles):
        show_values_on_bars(ax, h_v="h", space=0.4)
        ax.set_title(f"- {t} -", size=20, weight="bold")
        ax.set_xlabel("Frequency", weight="bold")
        ax.set_ylabel(f"{t}", weight="bold")
        ax.get_xaxis().set_ticks([])

    sns.despine()
    fig.tight_layout()
    del data, fig, ax, ax2, ax1, ax3, t, titles
    if cfg.show_figures:
        plt.show()

    """
    - Frequency between Days -
    Average Day per Case:
        -Most of the cases were taken on average between day10 and day20
        -There are also many cases took in day0 and day1
    Total Days/Case:
        -The total number of unique days per case varies between 1 and 6
        -The majority of the cases have data taken for ~ 3 and 5 days in total.
    """

    # 🚀 Image/Pixel width & high:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 15))
    titles = ["Img Width", "Img Height", "Pixel Width", "Pixel Height"]

    dt = train[train["image_width"] != 0.0].reset_index(drop=True)

    sns.barplot(data=dt["image_width"].value_counts().reset_index(),
                x="index", y="image_width", ax=ax1)

    sns.barplot(data=dt["image_height"].value_counts().reset_index(),
                x="index", y="image_height", ax=ax2, )

    sns.barplot(data=dt["pixel_width"].value_counts().reset_index(),
                x="index", y="pixel_width", ax=ax3, )

    sns.barplot(data=dt["pixel_height"].value_counts().reset_index(),
                x="index", y="pixel_height", ax=ax4)

    for ax, t in zip([ax1, ax2, ax3, ax4], titles):
        show_values_on_bars(ax, h_v="v", space=0.4)
        ax.set_title(f"- {t} -", size=20, weight="bold")
        ax.set_ylabel("Frequency", weight="bold")
        ax.set_xlabel(f"{t}", weight="bold")
        ax.get_yaxis().set_ticks([])

    sns.despine(left=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    del dt, fig, ax, ax2, ax1, ax3, ax4, t, titles
    if cfg.show_figures:
        plt.show()

    # 🐝 Log info to Dashboard
    if cfg.wandb_log:
        dt = train["image_width"].value_counts().reset_index()
        create_wandb_plot(x_data=dt["index"],
                          y_data=dt["image_width"],
                          x_name="Image Width/Height",
                          y_name="Frequency",
                          title="Image Width x Height",
                          log="img_specs", plot="bar")

        dt = train["pixel_width"].value_counts().reset_index()
        create_wandb_plot(x_data=dt["index"],
                          y_data=dt["pixel_width"],
                          x_name="Pixel Width/Height",
                          y_name="Frequency",
                          title="Pixel Width x Height",
                          log="pixel_specs", plot="bar")
        del dt

    #  🚀️ Segmentation View
    segment_per_id = train.groupby("id")["segmentation"].count() \
        .reset_index()["segmentation"].value_counts().reset_index()

    segment_per_class = train.groupby("class")["segmentation"].count().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7))
    titles = ["How many available segmentations we have per case?",
              "Which class has the most segmentations?"]

    sns.barplot(data=segment_per_id,
                x="index", y="segmentation", ax=ax1)

    sns.barplot(data=segment_per_class,
                x="class", y="segmentation", ax=ax2)

    for ax, t, x in zip([ax1, ax2], titles, ["no. segmentations per ID", "class"]):
        show_values_on_bars(ax, h_v="v", space=0.4)
        ax.set_title(f"- {t} -", size=20, weight="bold")
        ax.set_ylabel("Frequency", weight="bold")
        ax.set_xlabel(f"{x}", weight="bold")
        ax.get_yaxis().set_ticks([])

    sns.despine(left=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    del segment_per_class, segment_per_id, ax, ax2, ax1, x, t, fig
    if cfg.show_figures:
        plt.show()

    # 🐝 Log info to Dashboard
    if cfg.wandb_log:
        segment_per_id = train.groupby("id")["segmentation"].count() \
            .reset_index()["segmentation"].value_counts().reset_index()
        create_wandb_plot(x_data=segment_per_id["index"],
                          y_data=segment_per_id["segmentation"],
                          x_name="no. segmentations per ID",
                          y_name="Frequency",
                          title=f"{titles[0]}",
                          log="segm_id", plot="bar")

        segment_per_class = train.groupby("class")["segmentation"].count().reset_index()
        create_wandb_plot(x_data=segment_per_class["class"],
                          y_data=segment_per_class["segmentation"],
                          x_name="class",
                          y_name="Frequency",
                          title=f"{titles[1]}",
                          log="segm_class", plot="bar")
        del segment_per_class, segment_per_id, titles
        # 🐝 End Experiment
        wandb.finish()
        # 🐝 Save train.csv as artifact
        save_dataset_artifact(run_name="save_train",
                              artifact_name="train",
                              path="/home/titoare/Documents/ds/image_segmentation/input/train_w.csv",
                              cfg_wandb_config=cfg.wandb_config)

    """ 🤫 Experiment 2: Explore png images and their mask. """
    #  🚀️ Case image view
    if cfg.wandb_log:
        run = wandb.init(project='AWMadison', name='make_masks', config=cfg.wandb_config)

    CASE = "case123"

    # Sample a few images from speciffied case
    sample_paths = train[(train["segmentation"].isna() == False) & (train["case"] == CASE)]["path"] \
                       .reset_index().groupby("path")["index"].count() \
                       .reset_index().loc[:9, "path"].tolist()

    show_simple_images(sample_paths)

    #  🚀️ Segmentation / Mask View
    # "large_bowel", "small_bowel", "stomach"
    CASE = "case131_day0_slice_0066"
    img_path = train[train.id == CASE].path.values[0]
    mask_path = train[train.id == CASE].path_mask.values[0]
    mask = np.load(mask_path)
    # mask = get_id_mask(train, CASE) * 255
    img = read_image(img_path)
    show_img(img, mask)

    # Get cases with segmentation values.
    data = train.query(f"case == 'case123' & day == 'day20'")
    data = data[(data["segmentation"].apply(lambda x: type(x)) != float)]
    data = data.drop_duplicates(subset=['id'])
    img_paths = data.path.values
    mask_paths = data.path_mask.values
    mask = [np.load(path) for path in mask_paths]
    show_simple_images(img_paths[10:20], mask[10:20])
    show_simple_images(img_paths[20:30], mask[20:30])
    show_simple_images(img_paths[30:40], mask[30:40])
    show_simple_images(img_paths[40:50], mask[40:50])


