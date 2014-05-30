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
    print 'Segmentation of test image:'
    image = Image.open('flower.jpg')
    rgb_image = numpy.asarray(image, dtype='float32')
    orgb_image = SightSpotUtil.eval_orgb_image(rgb_image)
    segmentation_map = SightSpotUtil.eval_slic_map(orgb_image, 10.0, 0.1, 16)

