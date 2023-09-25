
from datasets import dataset_dict
import copy
from torch.utils.data import DataLoader, Subset

def setup_dataloader(hparams):
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir}
    # import pdb; pdb.set_trace()
    if hparams.dataset_name == 'phototourism':
        kwargs['img_downscale'] = hparams.img_downscale
        kwargs['val_num'] = hparams.num_gpus
        kwargs['use_cache'] = hparams.use_cache
        kwargs['use_mask'] = hparams.use_mask
    elif hparams.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(hparams.img_wh)
        kwargs['perturbation'] = hparams.data_perturb
        kwargs['random_occ'] = not hparams.nonrandom_occ
        kwargs['occ_yaw'] = hparams.occ_yaw
        kwargs['yaw_threshold'] = hparams.yaw_threshold
        kwargs['all_img_occ'] = hparams.all_img_occ
    
    train_dataset = dataset(split='train', **kwargs)
#     full_dataset = dataset(split='train', **kwargs)

    
#     train_datasets = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))
    img_sample_size = hparams.img_wh[0] * hparams.img_wh[1]
#     import pdb; pdb.set_trace()
    if hparams.public_dataset:
        kwargs_public = copy.deepcopy(kwargs)
#         kwargs_public['random_occ'] = True 
        kwargs_public['root_dir'] = hparams.public_root_dir
        public_dataset = dataset(split='train',**kwargs_public)
        public_train_ray_idx = []
        for ind in range(0,100,5):
            public_train_ray_idx.extend(list(range(ind*img_sample_size,(ind+1)*img_sample_size)))    
        public_train_dataset = Subset(public_dataset,public_train_ray_idx)
        public_train_loaders = DataLoader(public_train_dataset,shuffle=True,
                              num_workers=4,
                              batch_size=hparams.batch_size,
                              pin_memory=True)
#         import pdb; pdb.set_trace()
#         full_idx = set(list(range(img_sample_size*100)))
#         public_idx = set(public_train_ray_idx)
#         remaining_idx = list(full_idx-public_idx)
#         remaining_idx = [x for x in full_idx if x not in public_train_ray_idx]
        train_dataset_remaining = train_dataset
#         train_dataset_remaining = Subset(train_dataset,remaining_idx)
    else:
        train_dataset_remaining = train_dataset
        public_train_loaders = None
    
    #splitting the dataset
    partition_size = len(train_dataset_remaining) // hparams.num_clients
    lengths = [partition_size] * (hparams.num_clients)
    
#     import pdb; pdb.set_trace()
    train_datasets = []
    for ind in range(hparams.num_clients):
        train_datasets.append(Subset(train_dataset_remaining,range(ind*partition_size,ind*partition_size+partition_size)))
    val_dataset = dataset(split='val', **kwargs)
    
    train_loaders = []
    val_loaders = []
    
    for trainset in train_datasets:
        train_loaders.append(DataLoader(trainset,shuffle=True,
                          num_workers=4,
                          batch_size=hparams.batch_size,
                          pin_memory=True))
        val_loaders.append(DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True))
        
    
    

    return train_loaders, val_loaders, train_dataset, public_train_loaders