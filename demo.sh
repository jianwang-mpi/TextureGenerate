#!/usr/bin/env bash
model_path=pretrained_model/pretrained_weight.pkl

rm -r example_results/texture
mkdir example_results/texture
# set the gpu id to 2
python demo.py -g 2 -m ${model_path} -i example_results/input -o example_results/texture
