
from typing import Dict, List, Tuple

import torch
import flwr as fl
from flwr.common import NDArrays, Scalar
import glob

from models.nerf import *
from collections import OrderedDict
import os
# optimizer, scheduler, visualization
from utils import *
# load training and testing
from engine_pyt import *




class FlowerClient(fl.client.NumPyClient):
    def __init__(self, hparams, trainloader, vallodaer,full_dataset,client_number) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = vallodaer
        self.full_dataset = full_dataset
        self.hparams = hparams
        self.model = Net(self.hparams,self.full_dataset)
        self.client_num = client_number
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # figure out Transient params
        mydict = self.model.state_dict().keys()
#         avoid_transient = ['transient_rgb','transient_encoding','transient_sigma','transient_beta','embedding_t']
        mystr_arr = ['nerf_fine.transient_','embedding_t']
        transient_keys = []
#         import pdb; pdb.set_trace()
        for mystr in mystr_arr:
#             avoid_keys.append(key.startswith(mystr) for key in mydict)
            for key in mydict:
                if key.startswith(mystr):
                    transient_keys.append(key)
        self.transient_keys = transient_keys
                    
        self.model.to(self.device)  # send model to device
        self.epoch_num =1

    def set_parameters(self, parameters):
        """With the model paramters received from the server,
        overwrite the uninitialise model in this class with them."""
#         import pdb; pdb.set_trace()
        
#         print(avoid_keys)
        mykeys = self.model.state_dict().keys()
#         import pdb;pdb.set_trace()
        mykeys = [key for key in mykeys if key != 'embedding_t.weight'] #embedding_t is non learnable
        params_dict = zip(mykeys, parameters)
        state_dict_static = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if k not in self.transient_keys })
        ckpt_file_dir = os.path.join(f'ckpts/{self.hparams.exp_name}/clients/client_{self.client_num:0>2d}/**')
        files = glob.glob(ckpt_file_dir)
        if len(files)>0:
            
            ckpt_file = max(files)
#             print("loading from saved ckpt: ",ckpt_file)
            state_dict_transient = torch.load(ckpt_file)
            state_dict_transient =  OrderedDict({k: v for k, v in state_dict_transient.items() if k in self.transient_keys })

            state_dict_static.update(state_dict_transient)
        
        self.model.load_state_dict(state_dict_static, strict=False)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and conver them to a list of
        NumPy arryas. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for key, val in self.model.state_dict().items() if key not in self.transient_keys]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""
        
        # read from config
        server_round_, lr, epochs = config["round"], config["lr"], config["epochs"]
        
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # Define the optimizer and schedulter
        optim = get_optimizer(self.hparams, self.model.models_to_train)
        scheduler = get_scheduler(self.hparams, optim)
        
        if server_round_==1:
            train(self.model, self.trainloader, optim, scheduler, epochs=5, device=self.device,fix_static=True,fix_epochs=5)            
        else:
        # do local training
            train(self.model, self.trainloader, optim, scheduler, epochs=epochs, device=self.device)
                
        state_dict_all = self.model.state_dict()
        
        if self.hparams.encode_t:
            state_dict_transient = OrderedDict({k: v for k, v in state_dict_all.items() if k in self.transient_keys })
            ckpt_file = os.path.join(f'ckpts/{self.hparams.exp_name}/clients/client_{self.client_num:0>2d}/epoch_{server_round_:0>2d}.pth')
            torch.save(state_dict_transient,ckpt_file)
        
        state_dict_static = OrderedDict({k: v for k, v in state_dict_all.items() if k not in self.transient_keys })        
        ckpt_file_static = os.path.join(f'ckpts/{self.hparams.exp_name}/clients-static/client_{self.client_num:0>2d}/epoch_{server_round_:0>2d}.pth')        
        torch.save(state_dict_static,ckpt_file_static)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"val_psnr": accuracy}