
### 单GPU训练

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/MHAVSR/train_MHAVSR_REDS.yml

### 分布式训练

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/MHAVSR/train_MHAVSR_REDS.yml --launcher pytorch

### Model and Result

>Pre-trained models can be downloaded from baidu cloud.
>https://pan.baidu.com/s/1htgV3LxGqrGlwv9DdxrNBg
>(qwkx)
