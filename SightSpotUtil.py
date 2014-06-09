#!/usr/bin/env python

"""
This file contains implementation of few auxiliary algorithms for SightSpotDetector.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = "Copyright (c) 2014"
__license__   = "Apache 2.0"
__version__   = "1.0"

import math
import sys
import numpy
import numpy.linalg
import scipy.ndimage
import scipy.fftpack
from PIL import Image

def _eval_new_orgb_angle(rt, mask, add, mul):
    rt[mask] = add + mul * rt[mask]

def eval_orgb_image(rgb_image):
    """
    Converts specified PIL image to 3d NumPy array with oRGB components.

    Parameters
    ----------
    rgb_image : PIL.Image
        Should be 3 channel RGB image with 8bit per channels
    """
    assert(len(rgb_image.shape) == 3)
    assert(rgb_image.shape[-1] == 3)
    h, w = rgb_image.shape[0:2]
    orgb_array = numpy.reshape(numpy.array(rgb_image, dtype='float32') / 255.0, (h * w, 3))
    pre_transform = numpy.array([[0.2990, 0.5870, 0.1140], [0.8660, -0.8660, 0.0], [0.5000, 0.5000, -1.0000]])
    orgb_array = numpy.dot(pre_transform, orgb_array.transpose()).transpose()

    rg = orgb_array[:,1]
    yb = orgb_array[:,2]
    t = numpy.arctan2(rg, yb)
    mask_t_vs_0 = (t >= 0.0)
    mask_t_vs_pi_3 = (numpy.abs(t) >= math.pi / 3.0)
    rt = numpy.array(t)

    _eval_new_orgb_angle(rt, mask_t_vs_0 & mask_t_vs_pi_3, 0.25 * math.pi, 0.75)
    _eval_new_orgb_angle(rt, mask_t_vs_0 & ~mask_t_vs_pi_3, 0.0, 1.5)

    _eval_new_orgb_angle(rt, ~mask_t_vs_0 & mask_t_vs_pi_3, -0.25 * math.pi, 0.75)
    _eval_new_orgb_angle(rt, ~mask_t_vs_0 & ~mask_t_vs_pi_3, 0.0, 1.5)

    dt = rt - t
    cos_dt = numpy.cos(dt)
    sin_dt = numpy.sin(dt)
    r_yb = cos_dt * yb - sin_dt * rg
    r_rg = sin_dt * yb + cos_dt * rg

    orgb_array[:,1] = r_rg
    orgb_array[:,2] = r_yb
    orgb_image = orgb_array.reshape(h, w, 3)
    return orgb_image

################################################################################

def _ft_gaussian_blur(fft_image, sigma, pad_x, pad_y, w, h):
    convolved_image =  scipy.ndimage.fourier_gaussian(fft_image, [sigma, sigma, 0.0])
    blurred_image = scipy.fftpack.ifft2(convolved_image, axes=(0, 1)).real
    cut_image = blurred_image[pad_y:pad_y+h, pad_x:pad_x+w,:]
    return cut_image

def eval_saliency_map(orgb_image, small_sigma, large_sigma, params=(0.0, 0.0, 0.0)):
    """
    Calculates raw saliency map from given oRGB image.

    Parameters
    ----------
    orgb_image : ndarray
        Should be 3 channel oRGB image with float32 channels
    small_sigma : float
        Controls blur - it`s best to set ~ 1% of input image dimension.
    large_sigma : float
        Controls blur - it`s best to set ~ 20% of input image dimension.
    params : 3-tuple of floats or 'auto'
        Nonlinear correction is applied to saliency map to make it perceive better.

    Returns
    -------
    out : ndarray
        Raw saliency map of input image.
    """
    assert(len(orgb_image.shape) == 3)
    assert(orgb_image.shape[-1] == 3)
    assert(large_sigma > small_sigma)
    assert((type(params) == tuple and len(params) == 3) or params == 'auto')

    w = orgb_image.shape[1]
    h = orgb_image.shape[0]
    pad_x = w / 8
    pad_y = h / 8
    padded_image = numpy.zeros((h + 2 * pad_y, w + 2 * pad_x, 3), dtype='float32')
    padded_image[pad_y:pad_y+h, pad_x:pad_x+w,:] = orgb_image[:,:,:]
    fft_image = scipy.fftpack.fft2(padded_image, axes=(0, 1))

    small_blur_image_1 = _ft_gaussian_blur(fft_image, 1 * small_sigma, pad_x, pad_y, w, h)
    small_blur_image_2 = _ft_gaussian_blur(fft_image, 2 * small_sigma, pad_x, pad_y, w, h)
    small_blur_image_4 = _ft_gaussian_blur(fft_image, 4 * small_sigma, pad_x, pad_y, w, h)

    large_blur_image_1 = _ft_gaussian_blur(fft_image, 1 * large_sigma, pad_x, pad_y, w, h)
    large_blur_image_2 = _ft_gaussian_blur(fft_image, 2 * large_sigma, pad_x, pad_y, w, h)
    large_blur_image_4 = _ft_gaussian_blur(fft_image, 4 * large_sigma, pad_x, pad_y, w, h)

    difference_1 = small_blur_image_1 - large_blur_image_1
    difference_2 = small_blur_image_2 - large_blur_image_2
    difference_4 = small_blur_image_4 - large_blur_image_4

    saliency_map_1 = numpy.sqrt(numpy.sum(numpy.square(difference_1), axis=2))
    saliency_map_2 = numpy.sqrt(numpy.sum(numpy.square(difference_2), axis=2))
    saliency_map_4 = numpy.sqrt(numpy.sum(numpy.square(difference_4), axis=2))

    saliency_map = (saliency_map_1 + saliency_map_2 + saliency_map_4) / 9.0
    if params == 'auto':
        a = numpy.mean(saliency_map) / 2.0
        b = numpy.std(saliency_map)
        c = 1.0
    else:
        a = float(params[0])
        b = float(params[1])
        c = float(params[2])
    saliency_map_log = 1.0 / (1.0 + numpy.exp(-(saliency_map - b) / a))
    saliency_map_adj = (1.0 - c) * saliency_map + c * saliency_map_log
    return saliency_map_adj

################################################################################

def _get_lowest_grad_pos(orgb_image, x, y):
    nx, ny = int(x + 0.5), int(y + 0.5)
    low_x, high_x = nx - 2, nx+ 3
    assert (low_x >= 0)
    assert (high_x < orgb_image.shape[1])
    low_y, high_y = ny - 2, ny + 3
    assert (low_y >= 0)
    assert (high_y < orgb_image.shape[0])
    neighbor = orgb_image[low_y:high_y, low_x:high_x]
    opt_x, opt_y = nx, ny
    min_grad = sys.float_info.max
    for dy in xrange(1, 4):
        for dx in xrange(1, 4):
            diff_x = numpy.mean(neighbor[dy-1:dy+2, dx-1] - neighbor[dy-1:dy+2, dx+1], axis=0)
            diff_y = numpy.mean(neighbor[dy-1, dx-1:dx+2] - neighbor[dy+1, dx-1:dx+2], axis=0)
            grad = numpy.linalg.norm(diff_x, 2) + numpy.linalg.norm(diff_y, 2)
            if grad < min_grad:
                opt_x, opt_y = nx + dx - 2, ny + dy - 2
                min_grad = grad
    return opt_x, opt_y

def _init_clusters_centers(orgb_image, cell_size):
    width = orgb_image.shape[1]
    height = orgb_image.shape[0]
    cell_number_x = int(float(width) / cell_size - 0.5)
    cell_number_y = int(float(height) / cell_size - 0.5)
    pad_x = 0.5 * (width - cell_number_x * cell_size)
    pad_y = 0.5 * (height - cell_number_y * cell_size)
    assert(pad_x >= 1.0)
    assert(pad_y >= 1.0)
    cluster_centers = []
    for ny in xrange(cell_number_y + 1):
        y = ny * cell_size + pad_y
        for nx in xrange(cell_number_x + 1):
            x = nx * cell_size + pad_x
            lx, ly = _get_lowest_grad_pos(orgb_image, x, y)
            cluster_centers.append((lx, ly, orgb_image[ly, lx]))
    return cluster_centers

def _do_slic_iteration(orgb_image, cell_size, labels, distances, cluster_centers, alpha):
    width = orgb_image.shape[1]
    height = orgb_image.shape[0]
    coordinates = numpy.mgrid[0:height,0:width].swapaxes(0,2).swapaxes(0,1)

    for i in xrange(len(cluster_centers)):
        center_x, center_y = cluster_centers[i][0], cluster_centers[i][1]
        low_x, high_x = int(center_x - cell_size), int(center_x + cell_size + 1)
        low_y, high_y = int(center_y - cell_size), int(center_y + cell_size + 1)
        low_x = max(0, low_x)
        high_x = min(width, high_x)
        low_y = max(0, low_y)
        high_y = min(height, high_y)
        window = orgb_image[low_y:high_y, low_x:high_x]
        window_distances = distances[low_y:high_y, low_x:high_x]

        color_diff = window - cluster_centers[i][2]
        color_dist = numpy.sqrt(numpy.sum(numpy.square(color_diff), axis=2)) / 3.0
        mesh_x = numpy.square((numpy.arange(low_x, high_x) - center_x) / cell_size)
        mesh_y = numpy.square((numpy.arange(low_y, high_y) - center_y) / cell_size)
        mesh_xx, mesh_yy = numpy.meshgrid(mesh_x, mesh_y)
        coordinate_dist = numpy.sqrt(mesh_xx + mesh_yy)
        total_dist = (1.0 - alpha) * color_dist + alpha * coordinate_dist

        threshold_idx = (total_dist < window_distances)
        window_distances[threshold_idx] = total_dist[threshold_idx]

        labels[low_y:high_y, low_x:high_x][threshold_idx] = i
        distances[low_y:high_y, low_x:high_x] = window_distances

    for i in xrange(len(cluster_centers)):
        current_cluster_idx = (labels == i)
        cluster_size = numpy.sum(current_cluster_idx)
        if cluster_size != 0:
            new_coordinates = numpy.sum(coordinates[current_cluster_idx], axis=0)
            new_x = new_coordinates[1] / cluster_size
            new_y = new_coordinates[0] / cluster_size
            new_color = numpy.sum(orgb_image[current_cluster_idx], axis=0) / cluster_size
            cluster_centers[i] = (new_x, new_y, new_color)

def _extract_connected_components(labels):
    width = labels.shape[1] + 1
    height = labels.shape[0] + 1
    pad_labels = -2 * numpy.ones((height, width), dtype='int32')
    pad_labels[1:,1:] = labels[:,:]
    components = -1 * numpy.ones(pad_labels.shape, dtype='int32')

    component_counter = 0
    equality_classes = dict()
    for y in xrange(1, height):
        for x in xrange(1, width):
            center_label = pad_labels[y, x]
            left_label, top_label = pad_labels[y, x-1], pad_labels[y-1, x]
            left_is_neighbor, top_is_neighbor = (center_label == left_label), (center_label == top_label)
            if left_is_neighbor or top_is_neighbor:
                min_label = min(left_label, top_label)
                if left_is_neighbor and top_is_neighbor:
                    max_label = max(left_label, top_label)
                    if max_label not in equality_classes:
                        equality_classes[max_label] = set()
                    equality_classes[max_label].add(min_label)
                components[y, x] = min_label
            else:
                components[y, x] = component_counter
                component_counter += 1

    components = components[1:, 1:]
    number_of_components = numpy.max(components) + 1
    mapping = numpy.arange(0, number_of_components)
    for i in xrange(number_of_components):
        equality_class = i
        while equality_class in equality_classes:
            equality_class = min(equality_classes[equality_class])
        mapping[i] = equality_class
    components = mapping[components]
    assert(labels.shape == components.shape)
    return components

def _get_adj_matrix(components):
    width = components.shape[1]
    height = components.shape[0]
    number_of_components = numpy.max(components) + 1
    joint_clusters = []

    horizontal_one = components[:, 0:width-1]
    horizontal_two = components[:, 1:width]
    horizontal_idx = (horizontal_one != horizontal_two)

    horizontal_pairs = numpy.extract() # KEY!!!

    vertical_one = components[0:height-1, :]
    vertical_two = components[1:height, :]
    vertical_idx = (vertical_one != vertical_two)

    adj_matrix = dict()
    for i in xrange(number_of_components):
        adj_matrix[i] = set()

    return adj_matrix

def _find_broken_components(labels, components, cluster_centers):
    number_of_components = numpy.max(components) + 1
    is_broken = [True] * number_of_components
    for i in xrange(number_of_components):
        # KILL -1
        # SAVE THAT WHO HAS CLUSTER

    pass

def _enforce_connectivity(orgb_image, labels, cluster_centers):
    components = _extract_connected_components(labels)
    adj_matrix = _get_adj_matrix(components)
    broken_components = _find_broken_components(components, cluster_centers)

    #_reassign_it
    labels = components

def eval_slic_map(orgb_image, cell_size, alpha, iteration_number=16):
    """
    Calculates segmentation map of given ORGB image by SLIC algorithm.

    Parameters
    ----------
    orgb_image : ndarray
        Should be 3 channel oRGB image with float32 channels
    cell_size : int
        Controls number of clusters.
    alpha : float
        Define what is more important - cluster compactness or color similarity.
    iteration_number : int
        Number of iteration. Default value is 16.

    Returns
    -------
    out : ndarray
        Segmentation labels map for each pixel.
    """
    assert(cell_size >= 4)
    cluster_centers = _init_clusters_centers(orgb_image, cell_size)
    for k in xrange(iteration_number):
        labels = -1 * numpy.ones(orgb_image.shape[:2], dtype='int32')
        distances = sys.float_info.max * numpy.ones(orgb_image.shape[:2], dtype='float32')
        _do_slic_iteration(orgb_image, cell_size, labels, distances, cluster_centers, alpha)
    _enforce_connectivity(orgb_image, labels, cluster_centers)
    return labels

################################################################################

def _get_heatmap_palette():
    palette = []
    s = 1.0
    v = 255.0
    for idx in xrange(256):
        h = 240.0 * (1.0 - idx / 255.0)
        hi = math.floor(h / 60.0) % 6
        f =  (h / 60.0) - math.floor(h / 60.0)
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))
        aux_dict = {
            0: (v, t, p),
            1: (q, v, p),
            2: (p, v, t),
            3: (p, q, v),
            4: (t, p, v),
            5: (v, p, q)}
        r, g, b = aux_dict[hi]
        r = int(r + 0.5)
        g = int(g + 0.5)
        b = int(b + 0.5)
        palette.append((r, g, b))
    return palette

def eval_heatmap(saliency_map):
    """
    Calculates heatmap map from given saliency map.

    Parameters
    ----------
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.

    Returns
    -------
    out : PIL.Image
        Heatmap image.
    """
    assert(len(saliency_map.shape) == 2)
    palette = numpy.array(_get_heatmap_palette(), dtype='uint8')
    indices = numpy.array(255.0 * saliency_map + 0.5, dtype='uint8')
    heatmap = palette[indices]
    return Image.fromarray(heatmap)

################################################################################

#def eval

#def get_