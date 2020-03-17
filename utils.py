#!/usr/bin/env python
# coding=utf-8
'''
@Author: xieydd
@since: 2020-03-08 22:00:52
@lastTime: 2020-03-16 17:48:33
@LastAuthor: Do not edit
@message: 
'''

'''
@description: mxnet model to onnx
Attention: 
    #TODO onnx must use 1.3.0 https://github.com/apache/incubator-mxnet/issues/14589#issuecomment-479057849
    #TODO mxnet 1.5 and 1.6 only support onnx==1.3(opset == 8), so when convert prelu and BN  to  onnx, onnx2trt will read 8 version opset , and it will not support prelu, cause error;

    site:
    - https://github.com/onnx/onnx-tensorrt/issues/157
@param {type} 
    mxnet_model_path: mxnet model dir path include json and param files
    onnx_model_path: output onnx model file path
@return: 
'''


def mxnet2onnx(mxnet_model_path, onnx_model_path):
    import numpy as np
    import onnx
    import mxnet as mx
    import os
    from mxnet.contrib import onnx as onnx_mxnet
    # Export MXNet model to ONNX format via MXNet's export_model API
    converted_onnx = onnx_mxnet.export_model(mxnet_model_path+'/r100slim-symbol.json',
                                             mxnet_model_path+'/r100slim-0001.params',
                                             [(1, 3, 112, 112)], np.float32, onnx_model_path)

    # Check that the newly created model is valid and meets ONNX specification.
    model_proto = onnx.load(converted_onnx)
    onnx.checker.check_model(model_proto)


def onnx2trt(onnx_file_path, engine_file_path):
    import onnx
    import onnx_tensorrt.backend as backend
    import numpy as np
    model = onnx.load(onnx_file_path)
    engine = backend.prepare(model, device='CUDA:0')
    if(engine):
        print('Completed creating Engine')
        with open(engine_file_path, "wb") as f:
            f.write(engine.builder.build_cuda_engine(
                engine.network).serialize())
            print('Engine saved')
    else:
        print('Error building engine')
        exit(1)
    #input_data = np.random.random(size=(1, 3, 112, 112)).astype(np.float32)
    img = cv2.imread(
        "/tmp/snap_tensorrt/tensorrt_snap/build/test/img/test1.jpg").transpose(2, 0, 1)
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    # he arrays can either be stored in row-major (the default for numpy arrays) or column-major (see this wiki page for more info)
    input_data = np.array(input_data, dtype=input_data.dtype, order='C')
    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)


if __name__ == '__main__':
    # mxnet2onnx('/tmp/snap_tensorrt/facerecognition-model',
    #            '/tmp/snap_tensorrt/r100slim.onnx')
    onnx2trt('/tmp/snap_tensorrt/r100slim.onnx',
             '/tmp/snap_tensorrt/r100slim.trt')
