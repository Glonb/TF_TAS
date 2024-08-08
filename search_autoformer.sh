#!/bin/bash
python3 search_autoformer.py --data-path /autodl-tmp --data-set CIFAR10 --num_workers 4  --indicator-name dss --gp \
--param-limits 23 --change_qkv --relative_position --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


