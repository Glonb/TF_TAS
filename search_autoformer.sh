#!/bin/bash
python3 -m --use_env search_autoformer.py --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


