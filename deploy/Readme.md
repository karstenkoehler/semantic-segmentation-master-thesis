# Docker Images for Inference
There are two models available for inference:
* `schiffflieger46/semantic-segmentation:unet-23D`
* `schiffflieger46/semantic-segmentation:densenet-67D`

### Usage
```
docker run \
    -v $(pwd)/images:/images \
    -v $(pwd)/predictions:/predictions \
    schiffflieger46/semantic-segmentation:<model_version>
```

Two volumes need to be mounted for inference. The `images` directory holds all images for which a prediction should be made. The `predictions` directory will contain the results after the container finished successfully.  

## Build instructions
`*.hdf5` files with the trained models are not included in the repository because of their size.

### U-Net 
```
cd unet
docker build --tag schiffflieger46/semantic-segmentation:unet-23D .
```

### FC-DenseNet
```
cd densenet
docker build --tag schiffflieger46/semantic-segmentation:densenet-67D .
```

