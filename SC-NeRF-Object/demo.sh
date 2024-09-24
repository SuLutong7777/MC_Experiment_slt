cd NeRF

python run_nerf.py \
    --config /home/sulutong/SCNeRF-blenderv2.0/NeRF/configs/real_worlds/stump.txt \
    --expname demo_stump_pca \
    --chunk 8192 \
    --N_rand 1024 \
    --camera_model pinhole_rot_noise_10k_rayo_rayd \
    --ray_loss_type proj_ray_dist \
    --multiplicative_noise True \
    --i_ray_dist_loss 10 \
    --grid_size 10 \
    --ray_dist_loss_weight 0.0001 \
    --N_iters 100001 \
    --use_custom_optim True \
    --ray_o_noise_scale 1e-3 \
    --ray_d_noise_scale 1e-3 \
    --non_linear_weight_decay 0.1 \
    --add_ie 20000 \
    --add_od 40000 \
    --add_prd 60000
