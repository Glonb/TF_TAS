#!/bin/bash
python3 search_autoformer.py --data-set CIFAR10 --num_workers 4  --indicator-name dss --gp --param-limits 9 \
--change_qkv --relative_position --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


