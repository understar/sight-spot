#!/usr/bin/env python

"""
Test class.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = 'Copyright (c) 2014'
__license__   = 'Apache 2.0'
__version__   = '1.0'

import numpy
from PIL import Image
import SightSpotUtil

if __name__ == '__main__':
    print 'Calculate saliency map for test image:'
    image = Image.open('flower.jpg')
    rgb_image = numpy.asarray(image, dtype='float32')
    orgb_image = SightSpotUtil.eval_orgb_image(rgb_image)
    saliency_map = SightSpotUtil.eval_saliency_map(orgb_image, 5.0, 150.0, 'auto')
    heatmap_image = SightSpotUtil.eval_heatmap(saliency_map)
    saliency_image = Image.fromarray(saliency_map * 255)
    image.show('Source image')
    saliency_image.show('Saliency image')
    heatmap_image.show('Heatmap')
