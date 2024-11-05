exp_name='triplane-compressed'

voxel_size=0.001
update_init_factor=4
appearance_dim=0
ratio=1
tri_resolution=16
feat_dim=50

ulimit -n 4096

# ./train.sh -d nerf_synthetic/chair -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}  --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# wait

# ./train.sh -d nerf_synthetic/drums -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# # sleep 20s
# wait
# ./train.sh -d nerf_synthetic/ficus -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# # sleep 20s
# wait
# ./train.sh -d nerf_synthetic/hotdog -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# # sleep 20s
# wait
# ./train.sh -d nerf_synthetic/lego -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim} & 
# # sleep 20s
# wait
# ./train.sh -d nerf_synthetic/materials -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# # sleep 20s
# wait
# ./train.sh -d nerf_synthetic/mic -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# # sleep 20s
# wait
./train.sh -d nerf_synthetic/ship -l ${exp_name}  --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --tri_resolution ${tri_resolution} --feat_dim ${feat_dim}& 
# sleep 20s