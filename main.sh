
# 测试边缘生成是否顺利
# cd src
# cd rigid
# cd edge
# python evaluate_edge.py --input_path /home/zhangzhenbang/storage/mult_work/nc_rigid/data --output_dir /home/zhangzhenbang/storage/mult_work/nc_code/test_edge 

# 测试rigid算法是否运行顺利
# cd src
# cd rigid
# python main.py --iters 5 \
#         --input_dir /home/zhangzhenbang/storage/mult_work/nc_rigid/data \
#         --input_mask /home/zhangzhenbang/storage/mult_work/nc_rigid/mask_border_rotate \
#         --output_dir /home/zhangzhenbang/storage/mult_work/nc_code/test_rigid  --use_ransac 1
        
# # 测试elastic small size的运行
# cd src
# cd elastic
# python single_process.py --input_dir /home/zhangzhenbang/storage/mult_work/nc_rigid/data\
#                         --output_dir /home/zhangzhenbang/storage/mult_work/nc_code/test_elastic_single\
#                         --model_path /home/zhangzhenbang/storage/mult_work/model/EMnet/seed_42_theta_18.0_lr_0.005_iter_3_fine_1/checkpoints/53.pth

# 测试elastic big size的运行
# cd src
# cd elastic
# python process_big.py --input_dir /home/zhangzhenbang/storage/mult_work/nc_rigid/data\
#                         --output_dir /home/zhangzhenbang/storage/mult_work/nc_code/test_elastic_big\
#                         --model_path /home/zhangzhenbang/storage/mult_work/model/EMnet/seed_42_theta_18.0_lr_0.005_iter_3/checkpoints/90.pth \
#                         --height 1024 --width 1024 --patch_sz 512 --overlap 10


# 测试模型是否可以正确训练
# cd src
# cd elastic
# python train.py --dataset openog --root_dataset /home/zhangzhenbang/storage/Cremi/openog_data/train_data