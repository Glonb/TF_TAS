#!/bin/bash
python3 search_autoformer.py --data-path /mnt/workspace --data-set CIFAR100 --num_workers 4  --indicator-name mine --gp \
--param-limits 23 --change_qkv --relative_position --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


