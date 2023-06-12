# CUDA_VISIBLE_DEVICE='0, 1, 2, 3' python -m torch.distributed.launch --nproc_per_node 4 main.py
python Big2SmallMain.py \
    --storage_folder 'exp11' \
    --local_rank 0 \
    --state 'train' \
    --dataset 'cifar100' \
    --projector "Transformer" \
    --apply_successor True \
    --optimizer 'adam' \
    --lr 0.0005 \
    --epoch 50 \
    --batch_size 128 \
    --temperature 16
