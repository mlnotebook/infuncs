# infuncs.py
### Functions for importing and processing (augmenting and cropping) images and segmentations into a custom TensorFlow/Keras generator in 3D with option for one-hot encoding and multi-channel masking of segmentations.

*Should be used with ```transforms.py```*

This script contains several functions for use with a TensorFlow/Keras Generator object. For example, in my own workflow, the keras ```model.fit_generator()``` function takes in two custom generators (one for training and one for validation) which both output an array of images and an array of labels.

My code assumes that we have a root-folder containing subfolders with unique names/IDs (mixture of training, validation and test sets). Each subfolder contains both an image and its segmentation. The naming _must be consistent_ in each subfolder.

The functions in ```infuncs.py``` read-in and process this list of IDs and labels and outputs images and labels. It can perform data augmentation and cropping to both images and segmentations all in 3D.

This set of functions includes:
* ```get_split```: take the list of IDs and labels (stored as .npy) to use and splits it into training, validation and test sets.
* ```get_batch```: this function calls all others (except for get_split). takes in list of IDs, loads and resamples the images, performs data augmentaion and/or cropping before returning the processed image data and labels.
* ```load_and_resample```: a small function for loading image/segmentation data and resampling it to given dimensions.
* ```get_masks```: splits a single segmentation into a multi channel segmentation with one class per channel. One-hot encoding is also optional.
* ```crop_batch```: will crop a batch of images to a region surrounding its corresponding segmentation including some margin.
* ```do_aug```: augmentation function performing random combinations of rotation, scaling (zoom), flipping, translation and intensity shifting. Each transformation has its own random parameters which can be changed.

Note: any augmenter can be implemented so long as it returns an array of images and labels to be yielded to the generator.
