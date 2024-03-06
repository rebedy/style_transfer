# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os
import math
from argparse import ArgumentParser

import numpy as np
from skimage.transform import resize
from PIL import Image

from styler import styler



def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
                        dest='content', help='content image', metavar='CONTENT', required=True)
    parser.add_argument('--styles',
                        dest='styles', nargs='+', help='one or more style images', metavar='STYLE', required=True)
    parser.add_argument('--output',
                        dest='output', help='output path', metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int, default=700,
                        dest='iterations', help='iterations (default %(default)s)', metavar='ITERATIONS')
    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='statistics printing frequency', metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
                        dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg', metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency', metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int, 
                        dest='width', help='output width', metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
                        dest='style_scales', nargs='+', help='one or more style scales', metavar='STYLE_SCALE')
    parser.add_argument('--pre-trained-network', default='imagenet-vgg-verydeep-19.mat',
                        dest='pre_trained_network', help='path to network parameters (default %(default)s)')
    parser.add_argument('--content-weight-blend', type=float, default=1,
                        dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
                        metavar='CONTENT_WEIGHT_BLEND')
    parser.add_argument('--content-weight', type=float, default=5e0,
                        dest='content_weight', help='content weight (default %(default)s)', metavar='CONTENT_WEIGHT')
    parser.add_argument('--style-weight', type=float, default=5e2,
                        dest='style_weight', help='style weight (default %(default)s)', metavar='STYLE_WEIGHT')
    parser.add_argument('--style-layer-weight-exp', type=float, default=1,
                        dest='style_layer_weight_exp',
                        help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)')
    parser.add_argument('--style-blend-weights', type=float,
                        dest='style_blend_weights', help='style blending weights', nargs='+', metavar='STYLE_BLEND_WEIGHTS')
    parser.add_argument('--total-variation-weight', type=float, default=1e2,
                        dest='total_variation_weight', help='total variation regularization weight (default %(default)s)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        dest='optimizer', help='optimizer: [adam, adagrad, adadelta]', metavar='OPTIMIZER')
    parser.add_argument('--learning-rate', type=float, default=1e1,
                        dest='learning_rate', help='learning rate (default %(default)s)', metavar='LEARNING_RATE')
    parser.add_argument('--initial', 
                        dest='initial', help='initial image', metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
                        dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
                        metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
                        dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling', default='max',
                        dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)', metavar='POOLING')
    return parser


class StyleTrasfer:
    def __init__(self, parser, options):
        self.parser = parser
        self.opt = options
  
    def style_transfer(self):
        
        if not os.path.isfile(self.opt.pre_trained_network):
            self.parser.error("Network %s does not exist. (Did you forget to download it?)" % self.opt.pre_trained_network)

        content_image = self.img_read(self.opt.content)
        style_images = [self.img_read(style) for style in self.opt.styles]

        width = self.opt.width
        if width is not None:
            new_shape = (int(math.floor(float(content_image.shape[0]) /
                    content_image.shape[1] * width)), width)
            content_image = resize(content_image, new_shape, anti_aliasing=True)
        target_shape = content_image.shape
        for i in range(len(style_images)):
            style_scale = 1.0
            if self.opt.style_scales is not None:
                style_scale = self.opt.style_scales[i]
            shape = style_scale * target_shape[1] / style_images[i].shape[1]
            style_images[i] = resize(style_images[i], (shape,shape), anti_aliasing=True)
        style_blend_weights = self.opt.style_blend_weights
        if style_blend_weights is None:
            # default is equal weights
            style_blend_weights = [1.0/len(style_images) for _ in style_images]
        else:
            total_blend_weight = sum(style_blend_weights)
            style_blend_weights = [weight/total_blend_weight
                                for weight in style_blend_weights]

        initial = self.opt.initial
        if initial is not None:
            initial = resize(self.img_read(initial), content_image.shape[:2], anti_aliasing=True)
            # Initial guess is specified, but not noiseblend - no noise should be blended
            if self.opt.initial_noiseblend is None:
                self.opt.initial_noiseblend = 0.0
        else:
            # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
            if self.opt.initial_noiseblend is None:
                self.opt.initial_noiseblend = 1.0
            if self.opt.initial_noiseblend < 1.0:
                initial = content_image

        if self.opt.checkpoint_output and "%s" not in self.opt.checkpoint_output:
            self.parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

        for iteration, image in styler(
            pre_trained_network=self.opt.pre_trained_network,
            initial=initial,
            initial_noiseblend=self.opt.initial_noiseblend,
            content=content_image,
            styles=style_images,
            preserve_colors=self.opt.preserve_colors,
            iterations=self.opt.iterations,
            content_weight=self.opt.content_weight,
            content_weight_blend=self.opt.content_weight_blend,
            style_weight=self.opt.style_weight,
            style_layer_weight_exp=self.opt.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            total_variation_weight=self.opt.total_variation_weight,
            optimizer=self.opt.optimizer,
            learning_rate=self.opt.learning_rate,
            pooling=self.opt.pooling,
            print_iterations=self.opt.print_iterations,
            checkpoint_iterations=self.opt.checkpoint_iterations
        ):
            output_file = None
            combined_rgb = image
            if iteration is not None:
                if self.opt.checkpoint_output:
                    output_file = self.opt.checkpoint_output % iteration
                    # self.img_save(output_file, combined_rgb)
            else:
                output_file = self.opt.output
                # self.img_save(output_file, combined_rgb)
            if output_file:
                self.img_save(output_file, combined_rgb)


    def img_read(self, path):
        pil_img = Image.open(path)
        img = np.array(pil_img).astype(np.float)
        if len(img.shape) == 2:
            # grayscale
            img = np.dstack((img,img,img))
        elif img.shape[2] == 4:
            # PNG with alpha channel
            img = img[:,:,:3]
        return img


    def img_save(self, path, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path, quality=95)


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()

    style_transfer = StyleTrasfer(parser, options)
    style_transfer.style_transfer()
 