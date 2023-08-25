python train_fl_pl.py \
   --dataset_name blender \
   --root_dir /home/zt16/code/priv-nerf/nerf-pytorch/data/NeRF_Data/nerf_synthetic/lego \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 5 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name lego_nerf_occ_2clients_central \
   --encode_t --beta_min 0.1 --data_perturb occ \
   --num_clients 2 --num_rounds 5