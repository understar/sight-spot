#!/usr/bin/env python

"""
SightSpotDetector is fully-functional visual attention framework.

It allows the following:
1) Calculation of saliency map for given image (grayscale or heatmap-colored).
2) Obtain pixel-precise saliency map of given image.
3) Extract main objects using specific algorithm.
4) Extract eye movement path.

Framework uses lazy calculation so it`s doing less worthless work.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = "Copyright (c) 2014"
__license__   = "Apache 2.0"
__version__   = "1.0"

import math
import numpy
import scipy.ndimage
from PIL import Image
import SightSpotUtil

class SightSpotDetector():

    # Global constants.
    _MINIMAL_GRAIN_AREA = 16.0
    _SMALL_BLUR_RATIO = 1.0
    _LARGE_BLUR_RATIO = 4.0

    def __init__(self, image, grain=0.0001):
        """
        Public constructor of SightSpotDetector.

        Parameters
        ----------
        image : PIL.Image
            Should be any 3 channel RGB image with 8bit channels.
        grain : floating-point value
            Value should be between 0.0 and 1.0.
        """
        if not image is Image:
            raise Exception('Argument "image" should be PIL.Image!')
        self._segmentation_needed = False
        (width, height) = image.size
        area = float(width * height)
        grain_area = area * grain
        if grain_area >= self._MINIMAL_GRAIN_AREA:
            self._segmentation_needed = True
            self._segment_number = int(area / grain_area + 0.5)
        self._small_blur_sigma = self._SMALL_BLUR_RATIO * math.sqrt(grain_area / math.pi)
        self._large_blur_sigma = self._LARGE_BLUR_RATIO * math.sqrt(grain_area / math.pi)
        self._rgb_image = numpy.asarray(image, dtype='float32')
        self._orgb_image = None
        self._saliency_map = None
        self._segmentation_map = None

    def _lazy_get_orgb_image(self):
        if self._orgb_image is None:
            self._orgb_image = SightSpotUtil.eval_orgb_image(self._rgb_image)
        return self._orgb_image

    def _lazy_get_saliency_map(self):
        orgb_image = self._get_orgb_image()
        if self._saliency_map is None:
            self._saliency_map = SightSpotUtil.eval_saliency_map(orgb_image)
        return self._saliency_map

    #def _segmentation_map

    def get_saliency_map(self, type='raw'):
        if type == 'raw':
            return self._get_saliency_map()
        if type == 'precise':
            saliency_map = self._get_saliency_map()

            return self.saliency_map
        if type == 'heatmap':
            #saliency_map = self.get_saliency_map('raw')
            pass
        raise Exception('Unknown "type" argument at get_saliency_map() : type = ' + type)

    def extract_main_objects(self, type='rects'):
        pass

    def extract_sight_path(self):
        saliency_map = self._get_saliency_map()


    """

if __name__ == "__main__":
    pass


