python train.py \
  --root_dir /home/zt16/code/priv-nerf/nerf_pl/data/brandenburg_gate/ --dataset_name phototourism \
  --img_downscale 1 --use_cache --N_importance 64 --N_samples 64 \
  --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
  --num_epochs 30 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name brandenburg_scale1_nerfw