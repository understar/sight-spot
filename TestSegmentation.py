#!/usr/bin/env python

"""
Test class.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = 'Copyright (c) 2014'
__license__   = 'Apache 2.0'
__version__   = '1.0'

import random
import time
import numpy
from PIL import Image
import SightSpotUtil

def _visualize_clusters(segmentation_map):
    cluster_number = numpy.max(segmentation_map) + 2
    palette_items = []
    for i in xrange(cluster_number):
        r = random.randrange(256)
        g = random.randrange(256)
        b = random.randrange(256)
        palette_items.append((r, g, b))
    palette = numpy.array(palette_items, dtype='uint8')
    visualization = palette[segmentation_map + 1]
    return Image.fromarray(visualization)

def _visualize_contours(rgb_image, segmentation_map, color):
    neighbors_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    neighbors_y = [-1, -1, -1, 0, 1, 1, 1, 0]
    width = segmentation_map.shape[1]
    height = segmentation_map.shape[0]
    contours = numpy.zeros((height, width), numpy.bool)
    for y in xrange(1, height - 1):
        for x in xrange(1, width - 1):
            for dx, dy in zip(neighbors_x, neighbors_y):
                nx, ny = x + dx, y + dy
                if segmentation_map[ny, nx] != segmentation_map[y, x]:
                    contours[y, x] = True
                    contours[ny, nx] = True
    visualization = numpy.array(rgb_image, 'uint8')
    visualization[contours] = color
    visualization.astype('uint8')
    return Image.fromarray(visualization)

def _visualize_averaging(rgb_image, segmentation_map):
    cluster_number = numpy.max(segmentation_map) + 2
    palette_items = []
    for i in xrange(cluster_number):
        cluster_idx = (segmentation_map == i - 1)
        cluster_size = numpy.sum(cluster_idx)
        if cluster_size == 0:
            palette_items.append((0, 0, 0))
        else:
            avg = numpy.sum(rgb_image[cluster_idx], axis=0) / cluster_size
            palette_items.append(tuple(avg))
    print  palette_items
    palette = numpy.array(palette_items, dtype='uint8')
    visualization = palette[segmentation_map + 1]
    return Image.fromarray(visualization)

if __name__ == '__main__':
    print 'Segmentation of test image:'
    image = Image.open('flower.jpg')
    rgb_image = numpy.asarray(image, dtype='float32')
    orgb_image = SightSpotUtil.eval_orgb_image(rgb_image)

    start = time.clock()
    segmentation_map = SightSpotUtil.eval_slic_map(orgb_image, 64.0, 0.25, 16)
    print 'Segmentation map extracted in', time.clock() - start, 'sec.'

    _visualize_clusters(segmentation_map).show('Clusters in random colors')
    _visualize_contours(rgb_image, segmentation_map, (255, 255, 0)).show('Cluster contours')
    _visualize_averaging(rgb_image, segmentation_map).show('Averaging')

    #import cProfile
    #cProfile.run('segmentation_map = SightSpotUtil.eval_slic_map(orgb_image, 32.0, 0.1, 16)')