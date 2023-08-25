import argparse
from datasets import PhototourismDataset
import numpy as np
import os
import pickle

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_mask', default=False, action="store_true",
                        help='use masked images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()
    masked_str = '_masked' if args.use_mask else ''
    os.makedirs(os.path.join(args.root_dir, f'cache{masked_str}'), exist_ok=True)
    print(f'Preparing cache for scale {args.img_downscale}...')
    print(f'Using masked images: {args.use_mask}')
    dataset = PhototourismDataset(args.root_dir, 'train', args.img_downscale, use_mask=args.use_mask)
    
    # save img ids
    with open(os.path.join(args.root_dir, f'cache{masked_str}/img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    # save img paths
    with open(os.path.join(args.root_dir, f'cache{masked_str}/image_paths.pkl'), 'wb') as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(args.root_dir, f'cache{masked_str}/Ks{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)
    # save scene points
    np.save(os.path.join(args.root_dir, f'cache{masked_str}/xyz_world.npy'),
            dataset.xyz_world)
    # save poses
    np.save(os.path.join(args.root_dir, f'cache{masked_str}/poses.npy'),
            dataset.poses)
    # save near and far bounds
    with open(os.path.join(args.root_dir, f'cache{masked_str}/nears.pkl'), 'wb') as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_dir, f'cache{masked_str}/fars.pkl'), 'wb') as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs
    np.save(os.path.join(args.root_dir, f'cache{masked_str}/rays{args.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.root_dir, f'cache{masked_str}/rgbs{args.img_downscale}.npy'),
            dataset.all_rgbs.numpy())
    print(f"Data cache saved to {os.path.join(args.root_dir, f'cache{masked_str}')} !")