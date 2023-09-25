python train.py \
   --dataset_name blender \
   --root_dir /home/zt16/code/priv-nerf/nerfw_pl_priv/data/lego/res800_360view_IID_vertical_random \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 20 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name lego_nerf_people_nonIID_try2
   # --ckpt_path /home/zt16/code/priv-nerf/nerfw_pl_priv/ckpts/lego_nerf_people_nonIID/epoch=9.ckpt

   # --encode_t --beta_min 0.1 \

   # /home/zt16/code/priv-nerf/nerfw_pl_priv/data/lego/res400_360view 

## Learning NERF with occlusion data
# python train.py \
#    --dataset_name blender \
#    --root_dir /home/zt16/code/priv-nerf/nerf-pytorch/data/NeRF_Data/nerf_synthetic/lego \
#    --N_importance 64 --img_wh 400 400 --noise_std 0 \
#    --num_epochs 10 --batch_size 1024 \
#    --optimizer adam --lr 5e-4 --lr_scheduler cosine \
#    --nonrandom_occ --yaw_threshold 50.0 \
#    --exp_name test_old_code \
#    --encode_t --beta_min 0.1 \
#    --data_perturb occ 