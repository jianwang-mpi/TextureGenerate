## Re-ID Supervised Texture Generation

This is the code repo for the paper Re-Identification Supervised Texture Generation (CVPR2019) [[PDF]](https://arxiv.org/pdf/1904.03385v1.pdf).

### Requirement

- Python 3.6
- Pytorch 0.4.1

Install other python packages via:
```bash
pip install -r requirements.txt
```

### Demo

- Download the [pretrained weight](https://drive.google.com/open?id=1XM1bXm029xVJy2sek1Bun96fTACo7c_Q)

- Set the "model_path" in demo.sh to the path of pretrained weight

- Put some pedestrian images to ```example_results/input```

- Run demo.sh
```bash
bash demo.sh
```
- Get the resulting textures from ```example_results/texture```

- Render the 3D human model with texture using our another repo:
[BlenderRender](https://github.com/yt4766269/BlenderRender)

### Train

1. Download datasets:
    - market-1501
    - SURREAL
    - CUHK-SYSU (for background)
    - PRW (for background)

1. Generate the rendering tensors with [RenderingTensorGenerator](https://github.com/yt4766269/RenderingTensorGenerator).

1. Get the pretrained [re-id network](https://drive.google.com/open?id=1XM1bXm029xVJy2sek1Bun96fTACo7c_Q).

2. Set all paths and parameters in config.py

3. start train 
```bash
bash train.sh
```

4. you will get the trained models in ```model_log_path```

### Citation

----------------
If you use this code for your research, please cite our paper.

```bibtex
@article{wang2019reidsupervised,
  title={Re-Identification Supervised Texture Generation},
  author={Jian, Wang and Yunshan, Zhong and Yachun, Li and Chi, Zhang and Yichen, Wei},
  journal={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
