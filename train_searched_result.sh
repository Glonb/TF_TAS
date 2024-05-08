#!/bin/bash
python3 train.py --data-set CIFAR10 --gp --change_qk --relative_position \
--mode retrain --model_type 'AUTOFORMER' --cfg './experiments/subnet_autoformer/TF_TAS-T.yaml' --output_dir './OUTPUT/sample'


