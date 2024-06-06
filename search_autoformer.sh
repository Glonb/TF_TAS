#!/bin/bash
python3 search_autoformer.py --data-set CIFAR10 --num_workers 4  --indicator-name dss \ 
--gp --change_qkv --relative_position --param-limits 7 \ 
--cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


