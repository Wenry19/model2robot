
import sys
import yaml
import torch
import os
import subprocess
from pathlib import Path
import hashlib
import random
import numpy as np

def find_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    return device

def set_seed(seed):

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print("Output path already exists:", output_path)
        sys.exit(1)

def read_config_file(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_config(config):
    output_config_path = os.path.join(config["output_path"],
                                      "config.yaml")
    with open(output_config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

def make_file_only_read(filepath):
    os.chmod(filepath, 0o444)

def make_directory_only_read(directory):
    directory = Path(directory)
    for path in directory.rglob("*"):
        if path.is_file():
            make_file_only_read(path)
        elif path.is_dir():
            os.chmod(path, 0o555)
    os.chmod(directory, 0o555)

def get_git_info():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()

    commit_short = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True
    ).strip()

    dirty = subprocess.call(
        ["git", "diff", "--quiet"]
    ) != 0

    return {
        "commit": commit,
        "commit_short": commit_short,
        "dirty": dirty,
    }

def sha256_directory(directory):
    h = hashlib.sha256()

    directory = Path(directory)

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative_path = path.relative_to(directory)

            h.update(str(relative_path).encode())
            h.update(b"\0")

            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)

            h.update(b"\0")

    return h.hexdigest()

def generate_experiment_manifest(started_at,
                                 finished_at,
                                 input_artifact_path,
                                 output_artifact_path,
                                 experiment_type,
                                 output_manifest_path):

    manifest_filename = f"{experiment_type}_{os.path.basename(output_artifact_path)}.yaml"
    if not os.path.exists(output_manifest_path):
        os.makedirs(output_manifest_path)
    manifest_path = os.path.join(output_manifest_path, manifest_filename)

    manifest = {"experiment":
                {"type": experiment_type,
                 "started_at": started_at,
                 "finished_at": finished_at},
                "git": get_git_info(),
                "artifacts":
                 {"input-sha256": sha256_directory(input_artifact_path)
                                    if input_artifact_path is not None else "none",
                  "output-sha256": sha256_directory(output_artifact_path)}}

    with open(manifest_path, "w") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    make_file_only_read(manifest_path)
