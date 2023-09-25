import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,4'

from opt import get_opts
import torch


# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# dataloader for FL
from datasets.utils_data_fl import *

# training and testing
from engine_pyt import *

import copy
from collections import OrderedDict

import flwr as fl
from fl_flower_utils.fl_utils import SaveModelStrategy, weighted_average, fit_config, get_evaluate_fn, generate_client_fn

import pickle






def main(hparams):

    NUM_CLIENTS = hparams.num_clients
    
    trainloaders,valloaders, full_dataset_central, public_train_loader = setup_dataloader(hparams)
    hparams_public = copy.deepcopy(hparams)
    hparams_central = copy.deepcopy(hparams)
    hparams_public.encode_t = True
    hparams_central.encode_t = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Public Model Training
    if hparams.train_public_model:
        model_public = Net(hparams,full_dataset_central).to(device)
        # # # Define the optimizer and schedulter
        optim = get_optimizer(hparams, model_public.models_to_train)
        scheduler = get_scheduler(hparams, optim)

        # # do local training
        train(model_public, public_train_loader, optim, scheduler, epochs=hparams.public_num_epochs, device=device)
        loss, accuracy = test(model_public, valloaders[0], device=device)
        print("Public Model Loss: ",loss)
        print("Public Model Accuracy: ",accuracy)

        _state_dict_all = model_public.state_dict()
        _mydict = model_public.state_dict().keys()

        _mystr_arr = ['nerf_fine.transient_','embedding_t']
        _transient_keys = []
        #         import pdb; pdb.set_trace()
        for mystr in _mystr_arr:
        #             avoid_keys.append(key.startswith(mystr) for key in mydict)
            for key in _mydict:
                if key.startswith(mystr):
                    _transient_keys.append(key)


        _state_dict_transient = OrderedDict({k: v for k, v in _state_dict_all.items() if k in _transient_keys })
        _state_dict_static = OrderedDict({k: v for k, v in _state_dict_all.items() if k not in _transient_keys })

        os.system(f'mkdir -p ckpts/{hparams.exp_name}')
        _ckpt_file = os.path.join(f'ckpts/{hparams.exp_name}/central_model_transient.pth')
        _ckpt_file_static = os.path.join(f'ckpts/{hparams.exp_name}/central_model_static.pth')
        torch.save(_state_dict_transient,_ckpt_file)
        torch.save(_state_dict_static,_ckpt_file_static)
        print("Saved public model to ",_ckpt_file)
    if hparams.public_dataset:
        public_ckpt_file_static = os.path.join(f'ckpts/{hparams.exp_name}/central_model_static.pth')
        public_static_state_dict = torch.load(public_ckpt_file_static)

        public_model_static_params = [val.cpu().numpy() for _,val in public_static_state_dict.items()]

    if not hparams.public_dataset:
        strategy = SaveModelStrategy(
            fraction_fit=hparams.fraction_fit, 
            fraction_evaluate=hparams.fraction_evaluate,
            min_fit_clients=hparams.min_fit_clients,  
            min_evaluate_clients=hparams.min_evaluate_clients, 
            min_available_clients=int(
                NUM_CLIENTS * hparams.min_available_clients
            ),  
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,  
            evaluate_fn=get_evaluate_fn(valloaders[0],hparams_central,full_dataset_central),  
        )
    else:
        print("Initial Parameters given")
        strategy = SaveModelStrategy(
            fraction_fit=hparams.fraction_fit, 
            fraction_evaluate=hparams.fraction_evaluate,
            min_fit_clients=hparams.min_fit_clients,  
            min_evaluate_clients=hparams.min_evaluate_clients, 
            min_available_clients=int(
                NUM_CLIENTS * hparams.min_available_clients
            ),  
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,  
            evaluate_fn=get_evaluate_fn(valloaders[0],hparams_central,full_dataset_central), 
            initial_parameters=fl.common.ndarrays_to_parameters(public_model_static_params)#NOTE: finish this
        )

    strategy.init_central_params(hparams_central,full_dataset_central)

    client_fn_callback = generate_client_fn(trainloaders, valloaders,hparams,full_dataset_central)

    
    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {"num_cpus": hparams.num_cpus_per_client, "num_gpus": hparams.frac_gpus_per_client}
    os.system(f'mkdir -p ckpts/{hparams.exp_name}/server')
    for i in range(NUM_CLIENTS):
        os.system(f'mkdir -p ckpts/{hparams.exp_name}/clients/client_{i:0>2d}')
        os.system(f'mkdir -p ckpts/{hparams.exp_name}/clients-static/client_{i:0>2d}')

    logger_filename = f'ckpts/{hparams.exp_name}.txt'
    fl.common.logger.configure(identifier="myFlowerExperiment", filename=logger_filename)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,  # a callback to construct a client
        num_clients=NUM_CLIENTS,  # total number of clients in the experiment
        config=fl.server.ServerConfig(num_rounds=hparams.num_rounds),  # let's run for 10 rounds
        strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
        client_resources=client_resources,
    )    

    # Save the history object to a file
    history_file = f'ckpts/history_{hparams.exp_name}.pkl'
    with open(history_file, "wb") as f:
        pickle.dump(history, f)



if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)