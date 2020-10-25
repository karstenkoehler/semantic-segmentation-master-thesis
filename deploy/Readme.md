# Docker Images for Inference
There are two models available for inference:
* `imageseg/terrain-segmentation:unet-23D`
* `imageseg/terrain-segmentation:densenet-67D`

### Usage
```
docker run \
    -v $(pwd)/images:/images \
    -v $(pwd)/predictions:/predictions \
    imageseg/terrain-segmentation:<model_version>
```

Two volumes need to be mounted for inference. The `images` directory holds all images for which a prediction should be made. The `predictions` directory will contain the results after the container finished successfully.  

## Build instructions
`*.hdf5` files with the trained models are not included in the repository because of their size.

### U-Net 
```
cd unet
docker build --tag imageseg/terrain-segmentation:unet-23D .
```

### FC-DenseNet
```
cd densenet
docker build --tag imageseg/terrain-segmentation:densenet-67D .
```

