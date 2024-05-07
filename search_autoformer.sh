#!/bin/bash
python3 search_autoformer.py --data-set CIFAR10 --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


