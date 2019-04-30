## Re-ID Supervised 3D Texture Generation

This is the internal code repo for 3D Texture Generation.

### Requirement

- Python 3.6+
- Pytorch 0.4.1+

Install other python packages via:
```bash
pip install -r requirements.txt
```

### Demo

- Download the [pretrained weight](https://drive.google.com/open?id=14DsUrAgjjHZ_WiMQ2WFez22QD5-nR7gu)

- Set the "model_path" in demo.sh to the path of pretrained weight

- Put some pedestrian images to ```example_results/input```

- Run demo.sh
```bash
bash demo.sh
```
- Get UV map from ```example_results/texture```

- Render the 3D human model with texture using our another repo:
[BlenderRender](https://github.com/yt4766269/BlenderRender)

### Train

TODO