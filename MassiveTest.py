#!/usr/bin/env python

"""
Detect thresholded saliency maps for all image in specified folder.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = 'Copyright (c) 2014'
__license__   = 'Apache 2.0'
__version__   = '1.0'

import os
from PIL import Image
import SightSpotDetector

INPUT_FOLDER = r'/home/tinnulion/PycharmProjects/sight-spot/set/source'
OUTPUT_FOLDER = r'/home/tinnulion/PycharmProjects/sight-spot/set/sight-spot'

for root, dirs, files in os.walk(INPUT_FOLDER, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        print 'Processing:', path
        detector = SightSpotDetector.SightSpotDetector(path)

        saliency_map = Image.fromarray(255.0 * detector.get_saliency_map(type='raw')).convert('RGB')
        saliency_map.save(OUTPUT_FOLDER + '/raw_saliency_map.' + name)

        #heatmap = detector.get_heatmap(source='raw')
        #heatmap.save(OUTPUT_FOLDER + '/raw_heatmap.' + name)

        foreground = detector.get_foreground(source='precise')
        foreground.save(OUTPUT_FOLDER + '/foreground.' + name)




