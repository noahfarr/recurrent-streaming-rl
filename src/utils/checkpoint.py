import os

import wandb
from flax import serialization

from src.utils.typing import PyTree


def load_policy(
    template: PyTree,
    run_id: str,
    project: str = "recurrent-streaming-rl",
    entity: str | None = None,
    version: str = "latest",
) -> PyTree:
    api = wandb.Api()
    path = f"{project}/model-{run_id}:{version}"
    if entity:
        path = f"{entity}/{path}"
    artifact = api.artifact(path)
    artifact_dir = artifact.download()
    with open(os.path.join(artifact_dir, "model.msgpack"), "rb") as f:
        return serialization.from_bytes(template, f.read())
