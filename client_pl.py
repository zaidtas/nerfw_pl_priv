import flwr as fl
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TestTubeLogger
from collections import OrderedDict
import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
from nerfw_fl import *
from torch.utils.data import DataLoader
from opt import get_opts

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, hparams, cid, model, train_loader, val_loader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}/{cid}',
                                               '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=-1)
        # self.logger = TestTubeLogger(save_dir="logs/{cid}}",
        #                     name=hparams.exp_name,
        #                     debug=False,
        #                     create_git_tag=False,
        #                     log_graph=False)
        

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return _get_parameters(self.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1,checkpoint_callback=self.checkpoint_callback) #,logger=self.logger)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(config={}), len(self.trainloader), {}

    #TODO: implement evaluate
    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)

    #     trainer = pl.Trainer()
    #     results = trainer.test(self.model, self.test_loader)
    #     loss = results[0]["test_loss"]

    #     return loss, 10000, {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = NeRFSystem(hparams=hparams).to(DEVICE)

    train_datasets, val_dataset = setup_dataloader(hparams)
    # Load data 
    train_loader = DataLoader(train_datasets[int(cid)],
                          shuffle=True,
                          num_workers=4,
                          batch_size=hparams.batch_size,
                          pin_memory=True)

    val_loader = DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    # Create a  single Flower client representing a single organization
    return FlowerClient(hparams,cid,net, train_loader, val_loader)

#TODO: implement main
def main(hparams) -> None:
    # Create FedAvg strategy
    # TODO: implement evaluate metrics aggregation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 10 clients for training
        min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
        min_available_clients=2,  # Wait until all 10 clients are available
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    import pdb; pdb.set_trace()
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=hparams.num_clients,
        config=fl.server.ServerConfig(num_rounds=hparams.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    

if __name__ == "__main__":
    hparams = get_opts()
    main(hparams)