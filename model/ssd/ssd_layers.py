"""Some special pupropse layers for SSD."""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import math
from ssd.ssd_utils import BBoxUtility
#from scipy.misc import imresize
import cv2

class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
         4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        if K.common.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


class PriorBox(Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.

    # Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)

    # References
        https://arxiv.org/abs/1512.02325

    #TODO
        Add possibility not to have variances.
        Add Theano support
    """
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        if K.common.image_dim_ordering() == 'tf':
            self.waxis = 2
            self.haxis = 1
        else:
            self.waxis = 3
            self.haxis = 2
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        # define prior boxes shapes
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0)
        if K.backend() == 'tensorflow':
            pattern = [tf.shape(x)[0], 1, 1]
            prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)
        elif K.backend() == 'theano':
            #TODO
            pass
        return prior_boxes_tensor

class NMS(Layer):
    def __init__(self, num_classes, output_size = 19, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400, **kwargs):

        self.num_classes = num_classes
        self.output_size = output_size
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

        super(NMS, self).__init__(**kwargs)

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, x, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.01):

        mbox_loc = x[:, :, :4] # 4
        variances = x[:, :, -4:] # 4
        mbox_priorbox = x[:, :, -8:-4] # 4
        mbox_conf = x[:, :, 4:-8] # NUM_CLASSES

        results = []
        for i in range(self.num_samples):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                            mbox_priorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_size,self.output_size, self.num_classes)

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)

        if input_shape[0] is None:
            self.num_samples = 0
        else:
            self.num_samples = input_shape[0]

        if self.num_samples == 0:
            pref_mask = np.zeros((self.output_size, self.output_size, self.num_classes))
            pref_mask = tf.expand_dims(K.variable(pref_mask), 0)

            pattern = [tf.shape(x)[0], 1, 1, 1]
            pref_mask = tf.tile(pref_mask,pattern)
            return pref_mask

        else:
            results = self.detection_out(x)
            pref_mask = []
            for i in range(self.num_samples):
                det_label = results[i][:, 0]
                det_conf = results[i][:, 1]
                det_xmin = results[i][:, 2]
                det_ymin = results[i][:, 3]
                det_xmax = results[i][:, 4]
                det_ymax = results[i][:, 5]

                # Get detections with confidence higher than 0.6.
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

                top_conf = det_conf[top_indices] # num_boxes: top_conf.shape[0]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                pref_mask_mat = np.zeros((self.output_size, self.output_size, self.num_classes))
                for c in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[c] * self.output_size))
                    ymin = int(round(top_ymin[c] * self.output_size))
                    xmax = int(round(top_xmax[c] * self.output_size))
                    ymax = int(round(top_ymax[c] * self.output_size))
                    score = top_conf[c] # confidence score
                    label = int(top_label_indices[c])
                    pref_mask_mat[ymin:ymax, xmin:xmax, label-1] = score

                pref_mask.append(pref_mask_mat)

            pref_mask = tf.stack(pref_mask)
            return pref_mask

class Scat(Layer):
    """convert category to super category"""
    def __init__(self, scat2cat, pvec, **kwargs):
        self.num_scat = len(scat2cat.keys())
        self.scat2cat = scat2cat
        self.pvec = pvec
        super(Scat, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.num_scat)

    def get_config(self):
        config = {'scat2cat': self.scat2cat,
                    'pvec': self.pvec}
        return dict(list(config.items()))

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)

        if input_shape[0] is None:
            num_samples = 0
        else:
            num_samples = input_shape[0]

        if num_samples == 0:
            pref_mask = np.zeros((input_shape[1], input_shape[2], self.num_scat))
            pref_mask = tf.expand_dims(K.variable(pref_mask), 0)

            pattern = [tf.shape(x)[0], 1, 1, 1]
            pref_mask = tf.tile(pref_mask,pattern)
            return pref_mask
        else:
            pref_mask = []
            for i in range(self.num_samples):
                pref_mask_mat = []
                for sc in range(self.num_scat):
                    subtensor = [x[i, :, :, u] for u in scat2cat[sc]]
                    tmp = tf.stack(subtensor, axis=2)
                    tmp = tf.reshape(tmp,(input_shape[1],input_shape[2],len(scat2cat[sc])))
                    pref_mask_mat.append(tf.reduce_max(tmp, axis=2)*pvec[sc])
                pref_mask_mat = tf.stack(pref_mask_mat, axis=2)
                pref_mask.append(pref_mask_mat)       
            pref_mask = tf.stack(pref_mask)
            return pref_mask
        
class CentralBias(Layer):
    """add attention prior"""
    def __init__(self, cen_bias, **kwargs):
        self.cen_bias = cen_bias
        super(CentralBias, self).__init__(**kwargs)

    def get_config(self):
        config = {'cen_bias': self.cen_bias}
        return dict(list(config.items()))

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)

        if K.common.image_dim_ordering() == 'tf':
            self.waxis = 2
            self.haxis = 1
        else:
            self.waxis = 3
            self.haxis = 2

        cen_bias = cv2.resize(src=self.cen_bias, dsize=(input_shape[self.haxis],input_shape[self.waxis]))
        cen_bias = K.expand_dims(K.variable(cen_bias),0)
        cen_bias = K.expand_dims(cen_bias,3)
        pattern = [tf.shape(x)[0], 1, 1, 1]
        cen_bias = tf.tile(cen_bias,pattern)
        # TODO: binary? Normalization? Check the value range of the original tensor x
        result = tf.add(x, cen_bias)
        return result