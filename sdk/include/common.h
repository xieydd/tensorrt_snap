/*
 * @Author: xieydd
 * @since: 2020-03-11 16:52:56
 * @lastTime: 2020-03-14 22:36:05
 * @LastAuthor: Do not edit
 * @message: 
 */
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <dirent.h>
#include <half.h>
#include <algorithm>

namespace cudawrapper
{

class CudaStream
{
public:
    CudaStream()
    {
        cudaStreamCreate(&mStream);
    }

    operator cudaStream_t()
    {
        return mStream;
    }

    ~CudaStream()
    {
        cudaStreamDestroy(mStream);
    }

private:
    cudaStream_t mStream;
};

class CudaEvent
{
public:
    CudaEvent()
    {
        cudaEventCreate(&mEvent);
    }

    operator cudaEvent_t()
    {
        return mEvent;
    }

    ~CudaEvent()
    {
        cudaEventDestroy(mEvent);
    }

private:
    cudaEvent_t mEvent;
};

} // namespace cudawrapper

void softmax(std::vector<float>& tensor, int batchSize)
{
    size_t batchElements = tensor.size() / batchSize;

    for (int i = 0; i < batchSize; ++i)
    {
        float* batchVector = &tensor[i * batchElements];
        double maxValue = *std::max_element(batchVector, batchVector + batchElements);
        double expSum = std::accumulate(batchVector, batchVector + batchElements, 0.0, [=](double acc, float value) { return acc + exp(value - maxValue); });

        std::transform(batchVector, batchVector + batchElements, batchVector, [=](float input) { return static_cast<float>(std::exp(input - maxValue) / expSum); });
    }
}

void getFiles(const char* basePath, std::vector<std::string> &files)
{
    char path[1000];
    struct dirent *dp;
    DIR *dir = opendir(basePath);

    // Unable to open directory stream
    if (!dir)
        return;

    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {
            printf("%s\n", dp->d_name);

            // Construct new path from our base path
            strcpy(path, basePath);
            strcat(path, "/");
            strcat(path, dp->d_name);
            files.push_back(std::string(path));

            getFiles(path, files);
        }
    }

    closedir(dir);
}

static void bgr2rgb(const unsigned char* bgr, int w, int h,std::vector<float>& rgb, int totalElements)
{
    std::vector<float> c1(h*w);
    std::vector<float> c2(h*w);
    std::vector<float> c3(h*w);
    for (int j=0; j<h; j++)
    {
        for (int k = 0; k<w;k++)
        {
            c1[j*w+k] = static_cast<float>(bgr[2]);
            c2[j*w+k] = static_cast<float>(bgr[1]);
            c3[j*w+k] = static_cast<float>(bgr[0]);
            bgr+=3;
        }  
    }

    for (int j = 0;j<h*w;j++)
    {
        rgb[totalElements + j] = c1[j];
        rgb[totalElements + h*w + j] = c2[j];
        rgb[totalElements + 2*h*w + j] = c3[j];
    }
}

static void bgr2buffer(const unsigned char* bgr, int w, int h,std::vector<float>& buffer, int totalElements)
{
    std::vector<float> c1(h*w);
    std::vector<float> c2(h*w);
    std::vector<float> c3(h*w);
    for (int j=0; j<h; j++)
    {
        for (int k = 0; k<w;k++)
        {
            c1[j*w+k] = static_cast<float>(bgr[0]);
            c2[j*w+k] = static_cast<float>(bgr[1]);
            c3[j*w+k] = static_cast<float>(bgr[2]);
            bgr+=3;
        }  
    }

    for (int j = 0;j<h*w;j++)
    {
        buffer[totalElements + j] = c1[j];
        buffer[totalElements + h*w + j] = c2[j];
        buffer[totalElements + 2*h*w + j] = c3[j];
    }
}

// TODO
// Using warp logger class better

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

static int getBindingInputIndex(nvinfer1::IExecutionContext* context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
    }

using namespace nvinfer1;
class Logger: public ILogger
{
    public:
      Logger(Severity severity = Severity::kWARNING): reportableSeverity(severity) {}
      void log(Severity severity, const char* msg) override 
      {
        if (severity > reportableSeverity) return;
       switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOW: ";
            break;
       }
       std::cerr << msg << std::endl;
      }

      Severity reportableSeverity;
};
static Logger gLogger;

inline size_t volume(const Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<size_t>());
}
