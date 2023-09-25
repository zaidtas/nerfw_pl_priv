python train_fl_pytorch.py \
   --dataset_name blender \
   --root_dir /home/zt16/code/priv-nerf/nerfw_pl_priv/data/lego/res800_360view_IID_vertical_random \
   --public_root_dir /home/zt16/code/priv-nerf/nerfw_pl_priv/data/lego/res400_360view_random \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name lego_fl_test \
   --encode_t --beta_min 0.1 \
   --num_clients 20 --num_rounds 30 