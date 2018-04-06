# Image-Registration



## Dependencies

* numpy
* scipy
* matplotlib
* pillow
* scikit-image
* pytorch

## Dataset
Apple emoji dataset, which including ~2200 png images in size 160x160. Dataset source: https://github.com/iamcal/emoji-data

This is the preview of the dataset
![image1](https://github.com/limingwu8/Image-Registration/blob/master/images/dataset.png)

Deformation: I use this deformed algorithm on each images with different deformed parameters. Instead the following parameter -5 to 5, randomly generate number from -10 to 10 as parameters.
![image2](https://github.com/limingwu8/Image-Registration/blob/master/images/deformation_function.png)

Use the download raw image as image, deformed image as ground truth label.
![image3](https://github.com/limingwu8/Image-Registration/blob/master/images/deformed_img.png)


## Example of predicted images
![image1](https://github.com/limingwu8/UNet-pytorch/blob/master/images/prediction_results.png)
