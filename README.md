# CV lab

Consists of two tasks: stitching images and using CNN for planes and birds recognition.

## Stitching images

See [here](https://github.com/sofiia-tesliuk/stiching_images/blob/master/image_stitching.ipynb).


## Planes and birds recognition

For CNN used LeNet architecture. 

**Usage example:** 


Go to planes_birds directory: 
```
cd planes_birds
```

Train and save model:
```
python lenet_planes_birds.py --save-model 1 --weights pb_rmsp.hdf5
```

Load already saved model:
```
python lenet_planes_birds.py --load-model 1 --weights pb_rmsp.hdf5
```

Use SGD optimizer (default one is RMSProp):
```
python lenet_planes_birds.py --save-model 1 --weights pb_sgd.hdf5 --sgd-optimizer 1 
```

Use Adadelta optimizer (default one is RMSProp):
```
python lenet_planes_birds.py --save-model 1 --weights pb_adadelta.hdf5 --adadelta-optimizer 1 
```

Train on gray images:
```
python lenet_planes_birds.py --save-model 1 --weights pb_sgd.hdf5 --gray-scale 1 
```

Results:

Accuracy| RMSProp | SGD | Adadelta
------------- | ------------- | ------------- | -------------
Color image | 70.50% | **77.50%** | 76.30%
Gray image | 56.90% | 62.60% | 57.00%


