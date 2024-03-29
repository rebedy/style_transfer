# Neural style transfer with TensorFlow

This is a TensorFlow implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

## Requirements

### Data Files

- Pre-trained VGG-19 model: [VGG-19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)
- Content image: The image on which you want to apply the style
- Style image: The image from which you want to extract the style
- Output directory: The directory where the output image will be saved

### Python Packages

- TensorFlow
- NumPy
- Scikit-image
- Pillow
- SciPy

### Installation

To install the required packages, run the following command:

```bash
conda env create -f environment.yml
conda acrivate style_transfer
```

This will create a new conda environment called `style_transfer` and install the required packages. You can activate the environment by running `conda activate style_transfer`.

## Usage

To apply the style of the style image to the content image, run the following command:

```bash
python style_transfer.py --content imgs/content.jpg --styles imgs/style.jpg --output imgs/output.jpg
```

This will apply the style of the style image to the content image and save the result in the output directory.
This repository does not include the content image. You can use any image you want to apply the style to. The style image is included in the `imgs` directory.

## Example

You can use the following command to apply the style of the style image to the content image using the different optimizer [adam, adagrad, adadelta]:

```bash
python style_transfer.py --content imgs/content.jpg --styles imgs/style.jpg --output imgs/output_adadelta.jpg --optimizer adadelta
```

## Results

| <img src='imgs/style.jpg'> | <img src='imgs/output.jpg' width=50%> | 
|:--:|:--:|
| *Style* | *Output* |
