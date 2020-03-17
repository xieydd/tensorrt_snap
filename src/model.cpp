/*
 * @Author: xieydd
 * @since: 2020-03-11 16:35:00
 * @lastTime: 2020-03-14 23:03:04
 * @LastAuthor: Do not edit
 * @message: TRT Model Implament
 */
#include <model.h>
#include "assert.h"

Model::Model(Args &args)
{
    engine_filepath = args.dataDir + "/model/model.engine";
    img_dir = args.dataDir + "/data/";
    getFiles(img_dir.c_str(), img_files);

    if (img_files.size() == 0)
    {
        fprintf(stderr, "No data in %s for Inference\n", img_dir.c_str());
    }
    batch = args.batch;

    load_engine(engine_filepath.c_str());
}

int Model::load_engine(const char *file_path)
{
    unsigned char *buffer = NULL;
    int buffer_size = 0;

    FILE *fp = fopen(file_path, "rb");
    if (NULL == fp)
    {
        fprintf(stderr, "Read TRT Engine %s failed\n", file_path);
        fclose(fp);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    buffer_size = ftell(fp);
    if (buffer_size <= 0)
    {
        fprintf(stderr, "TRT Engine is empty\n");
        fclose(fp);
        return -1;
    }

    buffer = (unsigned char *)malloc(buffer_size);
    if (NULL == buffer)
    {
        fprintf(stderr, "Malloc memory for trt engine error\n");
        fclose(fp);
        return -1;
    }

    fseek(fp, 0, SEEK_SET);
    fread(buffer, 1, buffer_size, fp);
    fclose(fp);

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (NULL == runtime)
    {
        fprintf(stderr, "Create TRT Infer Runtime Error\n");
        return -1;
    }

    runtime->setDLACore(useDLACore);

    // TODO
    // nullptr means no plugin insert
    engine = runtime->deserializeCudaEngine(buffer, buffer_size, nullptr);
    if (NULL == engine)
    {
        fprintf(stderr, "Create TRT Engine Error\n");
        free(buffer);
        runtime->destroy();
        return -1;
    }

    // free buffer after generate engine
    free(buffer);

    context = engine->createExecutionContext();
    if (NULL == context)
    {
        fprintf(stderr, "Create TRT Engine Context Error\n");
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // Make sure num(input) + num(output) is fixed
    assert(engine->getNbBindings() == nums);
    if (engine->getNbBindings() != nums)
    {
        fprintf(stderr, "Bindings Data size is %d, and expected is %d\n", engine->getNbBindings(), nums);
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }
    // Make sure one is input
    assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

    // Create Host and Device buffers
    for (int i = 0; i < nums; ++i)
    {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t vol = static_cast<size_t>(batch);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        int vecDim = engine->getBindingVectorizedDim(i);
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = engine->getBindingComponentsPerElement(i);
            dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
            vol *= scalarsPerVec;
        }
        vol *= volume(dims);
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT:
            size[i] = vol * sizeof(float);
            break;
        case nvinfer1::DataType::kHALF:
            size[i] = vol * sizeof(half_float::half);
            break;
        case nvinfer1::DataType::kINT32:
            size[i] = vol * sizeof(int32_t);
            break;
        case nvinfer1::DataType::kINT8:
            size[i] = vol * sizeof(signed char);
            break;
        default:
            size[i] = vol * sizeof(float);
            break;
        }
        cudaMalloc(&bindings[i], size[i]);

        // TODO Only support float
        // Resize CPU buffers to fit Tensor.
        if (engine->bindingIsInput(i))
            inputTensor.resize(vol);
        else
            outputTensor.resize(vol);
    }
    return 0;
}

void Model::infer(cudawrapper::CudaStream stream)
{
    int inputId = getBindingInputIndex(context);
   cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size()* sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(batch, bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

Model::~Model()
{
    for (void *ptr : bindings)
        cudaFree(ptr);
}
