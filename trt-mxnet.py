#!/usr/bin/env python
# coding=utf-8
'''
@Author: xieydd
@since: 2020-03-13 16:08:23
@lastTime: 2020-03-16 16:54:30
@LastAuthor: Do not edit
@message: must pip install mxnet-tensorrt-cu92; https://beta.mxnet.io/guide/performance/backend/tensorRt.html
'''
import mxnet as mx
from mxnet.gluon.model_zoo import vision
import time
import os


def resnet_office():
    batch_shape = (1, 3, 224, 224)
    resnet18 = vision.resnet18_v2(pretrained=True)
    resnet18.hybridize()
    resnet18.forward(mx.nd.zeros(batch_shape))
    resnet18.export('resnet18_v2')
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet18_v2', 0)

    print('Building TensorRT engine')
    os.environ['MXNET_USE_TENSORRT'] = '1'
    arg_params.update(aux_params)
    all_params = dict([(k, v.as_in_context(mx.gpu(0)))
                       for k, v in arg_params.items()])
    executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(0), all_params=all_params,
                                                 data=batch_shape, grad_req='null', force_rebind=True)

    # Create sample input
    input = mx.nd.zeros(batch_shape)

    # Warmup
    print('Warming up TensorRT')
    for i in range(0, 10):
        y_gen = executor.forward(is_train=False, data=input)
        y_gen[0].wait_to_read()

    # Timing
    print('Starting TensorRT timed run')
    start = time.process_time()
    for i in range(0, 10000):
        y_gen = executor.forward(is_train=False, data=input)
        y_gen[0].wait_to_read()
    end = time.time()
    print(time.process_time() - start)


def resnet_ours():
    batch_shape = (1, 3, 112, 112)
    #resnet18 = vision.resnet18_v2(pretrained=True)
    # resnet18.hybridize()
    # resnet18.forward(mx.nd.zeros(batch_shape))
    # resnet18.export('resnet18_v2')
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        '/tmp/snap_tensorrt/facerecognition-model/r100slim', 1)

    print('Building TensorRT engine')
    os.environ['MXNET_USE_TENSORRT'] = '1'
    arg_params.update(aux_params)
    all_params = dict([(k, v.as_in_context(mx.gpu(0)))
                       for k, v in arg_params.items()])
    executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(0), all_params=all_params,
                                                 data=batch_shape, grad_req='null', force_rebind=True)

    # Create sample input
    #img = cv2.imread("/tmp/snap_tensorrt/tensorrt_snap/build/test/img/test1.jpg")
    #input = img.transpose(2,0,1)
    input = mx.nd.zeros(batch_shape)

    # Warmup
    print('Warming up TensorRT')
    for i in range(0, 10):
        y_gen = executor.forward(is_train=False, data=input)
        y_gen[0].wait_to_read()

    # Timing
    print('Starting TensorRT timed run')
    start = time.process_time()
    for i in range(0, 10000):
        y_gen = executor.forward(is_train=False, data=input)
        y_gen[0].wait_to_read()
    end = time.time()
    print(time.process_time() - start)


if __name__ == '__main__':
    resnet_ours()
