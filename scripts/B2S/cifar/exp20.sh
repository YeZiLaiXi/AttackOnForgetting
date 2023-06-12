# CUDA_VISIBLE_DEVICE='0, 1, 2, 3' python -m torch.distributed.launch --nproc_per_node 4 main.py
python Big2SmallMain.py \
    --storage_folder 'exp20' \
    --tea_arch_name 'ViT-B/32'\
    --stu_arch_name 'ResNet20'\
    --local_rank 0 \
    --state 'train' \
    --dataset 'cifar100' \
    --projector "Linear" \
    --apply_successor True \
    --optimizer 'adam' \
    --lr 0.0005 \
    --epoch 50 \
    --batch_size 64 \
    --temperature 16
