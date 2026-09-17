
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model2robot.utils as utils
from model2robot.training.trainer import Trainer
from model2robot.training.report import generate_report

from models.detectdoor import Detectdoor
from datasets.simple_dataset import SimpleDataset
from datasets.transforms import get_default_transform
from datasets.utils import get_images_paths_and_labels

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

def main():

    ### GET READY ###
    device = utils.find_device()
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)

    ### MODEL ###
    model = Detectdoor()
    model.to(device)

    ### LOSS FUNCTION ###
    loss_func = LOSS[config["training"]["loss"]]()

    ### OPTIMIZER ###
    optimizer = OPTIMIZER[config["training"]["optimizer"]](model.parameters(),
                                                           lr=config["training"]["learning_rate"])

    ### DATASETS ###

    train_img_paths, train_labels = get_images_paths_and_labels(config["dataset"]["train_path"],
                                                                config["dataset"]["class_names"])
    
    val_img_paths, val_labels = get_images_paths_and_labels(config["dataset"]["val_path"],
                                                            config["dataset"]["class_names"])

    transform = get_default_transform(input_height=config["model"]["input_height"],
                                      input_width=config["model"]["input_width"])

    train_dataset = SimpleDataset(img_paths=train_img_paths,
                                  labels=train_labels,
                                  transform=transform)
    val_dataset = SimpleDataset(img_paths=val_img_paths,
                                labels=val_labels,
                                transform=transform)
    # image, label = train_dataset[0]

    ### DATALOADER ###

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["dataloader"]["batch_size"],
                                  shuffle=config["dataloader"]["shuffle"],
                                  num_workers=config["dataloader"]["num_workers"],
                                  pin_memory=config["dataloader"]["pin_memory"],
                                  drop_last=config["dataloader"]["drop_last"])

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["dataloader"]["batch_size"],
                                shuffle=False,
                                num_workers=config["dataloader"]["num_workers"],
                                pin_memory=config["dataloader"]["pin_memory"],
                                drop_last=False)

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

if __name__ == "__main__":
    main()
