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

def _rgb2orgb(r, g, b):
    rgb_image = numpy.array([[[255.0 * r, 255.0 * g, 255.0 * b]]], dtype='float32')
    orgb = SightSpotUtil.eval_orgb_image(rgb_image)[0, 0]
    return tuple(orgb)

def _get_color(a, c_pos, c0, c_neg):
    assert (-1.0 <= a <= 1.0)
    if a >= 0.0:
        r = a * c_pos[0] + (1.0 - a) * c0[0]
        g = a * c_pos[1] + (1.0 - a) * c0[1]
        b = a * c_pos[2] + (1.0 - a) * c0[2]
    else:
        r = -a * c_neg[0] + (1.0 + a) * c0[0]
        g = -a * c_neg[1] + (1.0 + a) * c0[1]
        b = -a * c_neg[2] + (1.0 + a) * c0[2]
    assert(0.0 <= r <= 255.0)
    assert(0.0 <= g <= 255.0)
    assert(0.0 <= b <= 255.0)
    return [int(r + 0.5), int(g + 0.5), int(b + 0.5)]

def _colorize_channel(channel, c_pos, c_neg):
    h = channel.shape[0]
    w = channel.shape[1]
    color_array = numpy.zeros((h, w, 3), dtype='uint8')
    for y in xrange(h):
        for x in xrange(w):
            color_array[y, x] = _get_color(channel[y, x], c_pos, (0, 0, 0), c_neg)
    return Image.fromarray(color_array)

if __name__ == '__main__':
    print 'Test some basic oRGB color conversions:'
    print "RGB (0, 0, 0) is oRGB (%f, %f, %f)" % _rgb2orgb(0, 0, 0) # Black.
    print "RGB (1, 0, 0) is oRGB (%f, %f, %f)" % _rgb2orgb(1, 0, 0) # Red.
    print "RGB (0, 1, 0) is oRGB (%f, %f, %f)" % _rgb2orgb(0, 1, 0) # Green.
    print "RGB (0, 0, 1) is oRGB (%f, %f, %f)" % _rgb2orgb(0, 0, 1) # Blue.
    print "RGB (1, 1, 0) is oRGB (%f, %f, %f)" % _rgb2orgb(1, 1, 0) # Yellow.
    print "RGB (0, 1, 1) is oRGB (%f, %f, %f)" % _rgb2orgb(0, 1, 1) # Sky-blue.
    print "RGB (1, 0, 1) is oRGB (%f, %f, %f)" % _rgb2orgb(1, 0, 1) # Magenta.
    print "RGB (1, 1, 1) is oRGB (%f, %f, %f)" % _rgb2orgb(1, 1, 1) # White.

    print 'Show channels of test image:'
    image = Image.open('flower.jpg')
    rgb_image = numpy.asarray(image, dtype='float32')
    orgb_image = SightSpotUtil.eval_orgb_image(rgb_image)
    lu_channel = orgb_image[:,:,0]
    rg_channel = orgb_image[:,:,1]
    yb_channel = orgb_image[:,:,2]
    lu_image = Image.fromarray(lu_channel * 255 + 0.5)
    rg_image = _colorize_channel(rg_channel, (255, 0, 0), (0, 255, 0))
    yb_image = _colorize_channel(yb_channel, (255, 255, 0), (0, 0, 255))
    image.show('Source image')
    lu_image.show('Luminance channel')
    rg_image.show('Red-green channel')
    yb_image.show('Yellow-blue channel')

