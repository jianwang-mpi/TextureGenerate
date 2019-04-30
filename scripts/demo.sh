#!/usr/bin/env bash
model_path=/unsullied/sharefs/zhongyunshan/isilon-home/model-parameters/Texture/PCB_PerLoss2018-10-23_18:16:59.216650/2018-10-24_13:27:16.867817_epoch_120

rm -r ../example_results/rendered
rm -r ../example_results/texture
mkdir ../example_results/texture

rlaunch --cpu=2 --gpu=1 --memory=15000 -- python3 ../demo.py -m ${model_path} -i ../example_results/input -o ../example_results/texture

python ../smpl/diff_renderer.py ../example_results/texture ../example_results/rendered

#echo 'create example_result_after.zip'
#zip -r ../example_result_after.zip ../example_result_after/
#echo 'create example_result.zip'
#zip -r ../example_result.zip ../example_result/
