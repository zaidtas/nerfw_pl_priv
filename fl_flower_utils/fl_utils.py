import torch
# train and test
from engine_pyt import *

from collections import OrderedDict

from models.nerf import *

import numpy as np
import flwr as fl
from flwr.common import Metrics

from typing import Callable, Union, Dict, List, Optional, Tuple
from fl_flower_utils.fl_client import FlowerClient
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


def get_evaluate_fn(testloader,hparams_central,full_dataset_central):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        central_model = Net(hparams_central,full_dataset_central)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        central_model.to(device)  # send model to device

#         print("global model evaluation started")
        # set parameters to the model
#         print(central_model.state_dict().keys())
        params_dict = zip(central_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        central_model.load_state_dict(state_dict, strict=True)
        
#         print("global model loaded")
        

        # call test
        loss, accuracy = test(central_model, testloader, device)
        return loss, {"val_psnr": accuracy}

    return evaluate_fn

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "round": server_round,
        "epochs": 1,  # Number of local epochs done by clients
        "lr": 0.01,  # Learning rate to use by clients during fit()
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["val_psnr"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"val_psnr": sum(accuracies) / sum(examples)}


def generate_client_fn(trainloaders, valloaders,hparams,full_dataset):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        return FlowerClient(
            hparams=hparams, trainloader=trainloaders[int(cid)], vallodaer=valloaders[int(cid)],full_dataset=full_dataset,client_number=int(cid)
        )

    return client_fn

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # import pdb; pdb.set_trace()
            model = Net(self.hparams_central,self.full_dataset_central)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(model.state_dict(), f"ckpts/{self.hparams_central.exp_name}/server/round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics
    
    def init_central_params(self,hparams_central,full_dataset_central):
        self.hparams_central = hparams_central
        self.full_dataset_central = full_dataset_central