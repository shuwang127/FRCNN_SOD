#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Written by Shu Wang
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', 'car')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                     'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel'), 
        'vgg_cnn_m_1024_timely': ('VGG_CNN_M_1024_timely',
                     'vgg_cnn_m_1024_fast_rcnn_timely_iter_40000.caffemodel'),}

def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',  'VideoFrame_Proposal',  image_name + '_boxes.mat')
    if not os.path.isfile(box_file):
        return
    # timely: loading the boxes variable in the .mat file
    obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',  'VideoFrame_Images', image_name + '.jpg')
    # timely: read the image.
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # show the image
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    # open a file with a pointer
    save_txt_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',  'VideoFrame_txt', image_name + '.txt')
    fp = open(save_txt_file,  'w')
    
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        # select the boxes with the scores larger than CONF_THRESH
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        # using the Non-Maximum Suppression algrithom
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        # if nothing be detected.
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            # add the patch and textbox
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], fill=False,
                                edgecolor='red', linewidth=3.5))
            ax.text(bbox[0], bbox[1] - 2,
                                '{:s} {:.3f}'.format(cls, score),
                                bbox=dict(facecolor='blue', alpha=0.5),
                                fontsize=14, color='white')
            # save data to the file
            fp.write(cls+':'+'\n')
            fp.write(str(score)+'\t')
            fp.write(str(bbox[0])+'\t\t')
            fp.write(str(bbox[1])+'\t\t')
            fp.write(str(bbox[2])+'\t\t')
            fp.write(str(bbox[3])+'\n')
            
    # timely : save the image
    ax.set_title('Detection Results', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    save_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',  'VideoFrame_Results', image_name + '.jpg')
    plt.savefig(save_file, format='jpg')
    # timely : save the file
    fp.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg_cnn_m_1024_timely]',
                        choices=NETS.keys(), default='vgg_cnn_m_1024_timely') # modified by timely

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()

    # load the caffe model test.prototxt
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    # load the trained caffe model
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'default', 'KakouTrain', 
                              NETS[args.demo_net][1])
    
    # judge if the caffe model exits
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    # select the cpu/gpu model by the alternative argument
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    # load the net hyper-parameters
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    for i in range(1, 5+1):
        file_str = str(i).zfill(6)
        print 'Demo for ' + cfg.ROOT_DIR + 'data/demo/VideoFrame_Images/' + file_str + '.jpg'
        demo(net, file_str, ('car',))
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    #plt.show()
