/*
 * @Author: xieydd
 * @since: 2020-03-14 11:11:00
 * @lastTime: 2020-03-14 23:04:23
 * @LastAuthor: Do not edit
 * @message: Test TRT Inference
 */
#include <model.h>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char **argv)
{
    Args args;
    bool argsOK = parseArgs(args, argc, argv);
    if (!argsOK)
    {
        printHelpInfo();
        return -1;
    }

    if (args.help)
    {
        printHelpInfo();
        return 0;
    }

    Model model(args);
    int rows = model.height;
    int cols = model.weight;

    fprintf(stdout, "Building and running a GPU inference engine\n");

    std::vector<std::string> filepaths = model.img_files;
    int batch = model.batch;
    for (int i = 0; i < filepaths.size();i++)
    {
        int start = i * batch;
        std::vector<float> &buffers = model.inputTensor;
        size_t totalElements = 0;
        cudawrapper::CudaEvent start_event;
        cudawrapper::CudaEvent end_event;
        cudawrapper::CudaStream stream;
        float totalTime = 0.0;
        int ITERATIONS = 1; // TODO lt 0 will cause ERROR: ../rtSafe/safeContext.cpp (133) - Cudnn Error in configure: 7 (CUDNN_STATUS_MAPPING_ERROR) Not Solved
        if ((i + 1) * batch > filepaths.size())
        {
            batch = filepaths.size() - start;
            if (batch == 0) break;
        }

        model.batch = batch;

        for (int j = start; j < start + batch; j++)
        {
            cv::Mat input = cv::imread(filepaths[j], 1);
            cv::Mat bgr(rows, cols, input.type());
            //cv::Mat rgb;
            resize(input, bgr, bgr.size(), 0, 0, cv::INTER_LINEAR);
            //cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            //unsigned char* rgb_data = rgb.data;
            bgr2rgb(bgr.data, cols, rows, buffers,totalElements);
            //bgr2buffer(bgr.data, cols, rows, buffers,totalElements);
            //for (int i = 0; i < 112*112*3;i++)
            //{
            //    fprintf(stdout,"%f - %d \n", model.inputTensor[i], bgr.data[i]);
            //}
            //printf("%zu\n", elements);
            totalElements += rows * cols * 3;
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i)
        {
            float elapsedTime;
            cudaEventRecord(start_event, stream);
            model.infer(stream);
            cudaEventRecord(end_event, stream);
            // Wait until the work is finished.
            cudaStreamSynchronize(stream);
            cudaEventSynchronize(end_event);
            cudaEventElapsedTime(&elapsedTime, start_event, end_event);
            totalTime += elapsedTime;
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

        //std::cout << "Inference batch size " << batch << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << std::endl;
        std::cout << "Inference batch size " << batch << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << std::endl;


        //std::vector<float> output = model.outputTensor;
        //for (int i = 0;i< output.size();i++) {
        //    fprintf(stdout, "%f\n",output[i]);
        //}

        //softmax(output, batch);
        //sort(output.rbegin(),output.rend());
        //int top_k = 10;
        //for (int i = 0;i< top_k;i++) {
        //    fprintf(stdout, "top %d : %f\n", i+1,output[i]);
        //}
    }
}
