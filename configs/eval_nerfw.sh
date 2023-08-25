python eval.py \
  --root_dir /home/zt16/code/priv-nerf/nerf_pl/data/brandenburg_gate/ \
  --dataset_name phototourism --scene_name brandenburg_test_scale1_ckp2 \
  --split test --N_samples 256 --N_importance 256 \
  --N_vocab 1500 --encode_a --encode_t \
  --ckpt_path ckpts/brandenburg_scale1_nerfw/epoch\=2.ckpt \
  --chunk 16384 --img_wh 640 480