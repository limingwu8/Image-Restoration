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
<p align="center">
	<img src="https://github.com/limingwu8/Image-Registration/blob/master/images/dataset.png">
</p>

Deformation: for each training image, a gaussian mixture overlapped by 3 randomly generated gaussian is applied to the image to perform a deformation. Random gaussian noise is also applied to each image.
<p align="center">
	<img src="https://github.com/limingwu8/Image-Registration/blob/master/images/deformation_algorithm.png">
</p>

Use the download raw image as image, deformed image as ground truth label.
<p align="center">
	<img src="https://github.com/limingwu8/Image-Registration/blob/master/images/deformed_img.png">
</p>

## Example of predicted images
<p align="center">
	<img src="https://github.com/limingwu8/Image-Registration/blob/master/images/prediction_results.png">
</p>