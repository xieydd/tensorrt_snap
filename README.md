<!--
 * @Author: xieydd
 * @since: 2020-03-08 22:18:23
 * @lastTime: 2020-03-17 11:37:42
 * @LastAuthor: Do not edit
 * @message: 
 -->
### TENSORRT_SNAP

A Demo of TensorRT Project

Snap project use tensorrt for inference.

lib and include need be downloaded in https://developer.nvidia.com/nvidia-tensorrt-7x-download, We use TensorRT Version 7;

Notice: Input is trt.engine, it is generated by onnx2trt, tha param below should be same;
                [-b max_batch_size (default 32)]
                [-w max_workspace_size_bytes (default 1 GiB)]
                [-d model_data_type_bit_depth] (32 => float32, 16 => float16)


Maybe Use cv::cuda:Mat or Deep Stream 多路并发


Site:
- https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/TensorRT-introduction/simpleOnnx.cpp