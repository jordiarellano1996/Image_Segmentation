import wandb


def create_wandb_plot(x_data=None, y_data=None, x_name=None, y_name=None, title=None, log=None, plot="line"):
    """
    Create and save lineplot/barplot in W&B Environment.
    x_data & y_data: Pandas Series containing x & y data
    x_name & y_name: strings containing axis names
    title: title of the graph
    log: string containing name of log
    """

    data = [[label, val] for (label, val) in zip(x_data, y_data)]
    table = wandb.Table(data=data, columns=[x_name, y_name])

    if plot == "line":
        wandb.log({log: wandb.plot.line(table, x_name, y_name, title=title)})
    elif plot == "bar":
        wandb.log({log: wandb.plot.bar(table, x_name, y_name, title=title)})
    elif plot == "scatter":
        wandb.log({log: wandb.plot.scatter(table, x_name, y_name, title=title)})


def create_wandb_hist(x_data=None, x_name=None, title=None, log=None):
    """
    Create and save histogram in W&B Environment.
    x_data: Pandas Series containing x values
    x_name: strings containing axis name
    title: title of the graph
    log: string containing name of log
    """

    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log: wandb.plot.histogram(table, x_name, title=title)})


def save_dataset_artifact(artifact_name, path):
    """
    Saves dataset to W&B Artifactory.
    project_name: name of project you are working on.
    run_name: name of the experiment
    artifact_name: under what name should the dataset be stored
    path: path to the dataset
    """

    artifact = wandb.Artifact(name=artifact_name,
                              type='dataset')
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    print("Artifact has been saved successfully.")