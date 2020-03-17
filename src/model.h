/*
 * @Author: xieydd
 * @since: 2020-03-11 16:12:28
 * @lastTime: 2020-03-14 22:56:20
 * @LastAuthor: Do not edit
 * @message: TRT Model Definition Head File
 */
#ifndef MODEL_H
#define MODEL_H
#include <string>
#include <common.h>
#include <argsParser.h>
#include <vector>

class Model
{
public:
    int load_engine(const char *file_path);
    void infer(cudawrapper::CudaStream stream);
    Model(Args &args);
    ~Model();
public:
    std::vector<float> inputTensor;
    std::vector<float> outputTensor;
    std::vector<std::string> img_files;

    int batch{1}; // Same like onnx2trt config
    int height = 112;
    int weight = 112;

private:
    std::string engine_filepath;
    std::string img_dir;
    nvinfer1::ICudaEngine *engine;
    //std::shared_ptr<nvinfer1::ICudaEngine> engine;
    nvinfer1::IRuntime *runtime;
    nvinfer1::IExecutionContext *context;
    enum
    {
        nums = 2
    }; // num(input) + num(output)

    int useDLACore{0};
    void *bindings[nums]{0};
    size_t size[nums];
};
#endif
