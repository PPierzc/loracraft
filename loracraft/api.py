import wandb
from PIL import Image

def log_images(run, images):
    run.log({"images": [wandb.Image(img) for img in images]})


def query_runs(query):
    """
    Query runs from the wandb API
    :param query: dict with query parameters
    :return:
    """
    # attach "config" to each key in query
    query = {f"config.{k}": v for k, v in query.items()}

    runs = wandb.Api().runs(f"ppierzc/loracraft", filters=query)
    return runs


def get_images_from_run(run):
    """
    Get the images from a run
    :param run: wandb run object
    :return: list of images
    """
    return [Image.open(img.download(exist_ok=True).name) for img in run.files() if img.name.endswith(".png")]