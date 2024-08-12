#!/bin/bash
python3 search_autoformer.py --data-path ~/autodl-tmp/NIHChestXRay/label_one --data-set NIHChestXRay --num_workers 8  --indicator-name dss --gp \
--param-limits 23 --change_qkv --relative_position --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


