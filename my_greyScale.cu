#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char **argv){

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Maximum threads per along x, y, z: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1]<<" " <<prop.maxThreadsDim[2]<<" "<<std::endl;    

    cv::Mat h_img = cv::imread("../cinque_terre_small.jpg");

    std::cout<<h_img.size()<<"\t"<<h_img.total()<<std::endl;


    const int IMG_SIZE_BYTES = h_img.rows * h_img.cols  * h_img.elemSize();  //h_img.elemSize() = number of bytes per pixel, including all channels
    std::cout<<IMG_SIZE_BYTES<<std::endl;

    uchar *d_in, *d_out;
    cudaMalloc( (void**) &d_in, IMG_SIZE_BYTES);
    cudaMalloc((void**) &d_out, IMG_SIZE_BYTES/3);  //    divide by 3 because black and white 

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}