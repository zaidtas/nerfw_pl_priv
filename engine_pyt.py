import torch
# metrics
from metrics import *

# losses
from losses import loss_dict

def train(net,trainloader,optimizer,scheduler,epochs,device,fix_static=True,fix_epochs=1,print_s=False):    
    criterion = loss_dict['nerfw'](coef=1)
    net.train()
    freeze_switch = False
    if fix_static:
#         print("freezing static wts")
        freeze_switch = True
        mydict = net.state_dict()
        mystr_arr = ['nerf_fine.transient_','embedding_t']
        freeze_keys = []
    #         import pdb; pdb.set_trace()
        for mystr in mystr_arr:
            for key in mydict:
                if key.startswith(mystr):
                    freeze_keys.append(key)
        for name, param in net.named_parameters():
            if param.requires_grad and name not in freeze_keys:
#                 print(name)
                param.requires_grad = False

    
    for ep in range(epochs):
        itera=0
        if (ep>=fix_epochs and freeze_switch) or ep==epochs-1:
#             print("freezing static wts")
            freeze_switch = False
            for name, param in net.named_parameters():
                    if not param.requires_grad and name not in freeze_keys:
#                         print(name)
                        param.requires_grad = True
        
        for batch in trainloader:

#             if itera>10:
#                 break
                
            rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
            rays = rays.to(device)
            rgbs = rgbs.to(device)
            ts = ts.to(device)
            optimizer.zero_grad()
            results = net(rays,ts)
            loss_d = criterion(results,rgbs)
            loss = sum(l for l in loss_d.values())
            loss.backward()
            optimizer.step()
            itera+=1
            
                
        scheduler.step()
        if print_s:
            print('iteration done: ',ep)
    return net

def test(net, valloader, device):
    criterion = loss_dict['nerfw'](coef=1)
    psnr_, loss = 0.0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in valloader:
#             batch.to(device)
            rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
            rays = rays.to(device)
            rgbs = rgbs.to(device)
            ts = ts.to(device)
            rays = rays.squeeze() # (H*W, 3)
            rgbs = rgbs.squeeze() # (H*W, 3)
            ts = ts.squeeze() # (H*W)
            results = net(rays, ts)
            loss_d = criterion(results, rgbs)
            loss += sum(l for l in loss_d.values())
            typ = 'fine' if 'rgb_fine' in results else 'coarse'

            psnr_ += psnr(results[f'rgb_{typ}'], rgbs)
    val_psnr = psnr_ / len(valloader.dataset)
            
    return loss, val_psnr