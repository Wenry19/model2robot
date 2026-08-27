
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from datasets.simple_dataset import SimpleDataset
from datasets.utils import get_images_paths_and_labels

from training.trainer import Trainer
from training.report import generate_report

LOSS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "BCELoss": nn.BCELoss,
}

OPTIMIZER = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
}

if __name__ == "__main__":

    ### GET READY ###
    device = utils.find_device()
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)

    ### MODEL ###
    model = utils.instantiate_class(config["model"]["module"],
                                    config["model"]["class"])
    model.to(device)

    ### LOSS FUNCTION ###
    loss_func = LOSS[config["training"]["loss"]]()

    ### OPTIMIZER ###
    optimizer = OPTIMIZER[config["training"]["optimizer"]](model.parameters(),
                                                           lr=config["training"]["learning_rate"])

    ### DATASETS ###

    target_path = config["dataset"]["train_path"]
    train_img_paths, train_labels = get_images_paths_and_labels(target_path,
                                                                config["dataset"]["class_names"])

    target_path = config["dataset"]["val_path"]
    val_img_paths, val_labels = get_images_paths_and_labels(target_path,
                                                            config["dataset"]["class_names"])

    input_width = config["model"]["input_width"]
    input_height = config["model"]["input_height"]
    transform = transforms.Compose([transforms.Resize((input_height, input_width)),
                                    transforms.ToTensor()])

    train_dataset = SimpleDataset(train_img_paths, train_labels, transform)
    val_dataset = SimpleDataset(val_img_paths, val_labels, transform)
    # image, label = train_dataset[0]

    ### DATALOADER ###

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["training"]["batch_size"],
                                  shuffle=config["dataloader"]["shuffle"],
                                  num_workers=config["dataloader"]["num_workers"],
                                  pin_memory=config["dataloader"]["pin_memory"],
                                  drop_last=config["dataloader"]["drop_last"])

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["training"]["batch_size"],
                                shuffle=False,
                                num_workers=config["dataloader"]["num_workers"],
                                pin_memory=config["dataloader"]["pin_memory"])

    ### TRAIN ###
    trainer = Trainer(model,
                      train_dataloader,
                      val_dataloader,
                      device,
                      loss_func,
                      optimizer,
                      config["output_path"],
                      config["seed"])

    trainer.train(config["training"]["epochs"])

    ### RESULTS ###

    generate_report(config["output_path"],
                    trainer.timestamps,
                    trainer.train_losses,
                    trainer.val_losses,
                    trainer.val_accuracies)
