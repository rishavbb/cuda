#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <iostream>


__global__ void black_white(uchar *d_out, uchar *d_in, int rows, int cols, int n_channels){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= rows || x >= cols)
        return;

    int pixel_idx = y * cols + x;  // since the entire 2D image is stored in 1D

    int rgbIdx = pixel_idx * n_channels;  // since there are n_channels

    uchar R = d_in[rgbIdx];
    uchar G = d_in[rgbIdx + 1];
    uchar B = d_in[rgbIdx + 2];

    d_out[pixel_idx] = .299f * R + .587f * G + .114f * B;

}


int main(int argc, char **argv){

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum threads per block for this system: " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Maximum threads per along x, y, z: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1]<<" " <<prop.maxThreadsDim[2]<<" "<<std::endl;    

    cv::Mat h_img = cv::imread("../cinque_terre_small.jpg");

    std::cout<<h_img.size()<<"\t" <<std::endl;


    const int IMG_SIZE_BYTES = h_img.rows * h_img.cols  * h_img.elemSize();  //h_img.elemSize() = number of bytes per pixel, including all channels
    std::cout<<"Size of color image: "<<IMG_SIZE_BYTES<<std::endl;

    cv::Mat bw_img(h_img.rows, h_img.cols, CV_8UC1);;

    uchar *d_in, *d_out;
    cudaMalloc( (void**) &d_in, IMG_SIZE_BYTES);
    cudaMalloc((void**) &d_out, IMG_SIZE_BYTES / h_img.channels());  //    divide because black and white is 1 channel

    cudaMemcpy(d_in, h_img.data, IMG_SIZE_BYTES, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);   // EACH BLOCK HAS 32*32 THREADS - I AM ASSUMING THERE ARE 3 CHANNELS ONLY - IF MORE I REDUCE BECAUSE MAX THREADS 1024
    dim3 gridSize((h_img.cols + blockSize.x - 1) / blockSize.x,
                  (h_img.rows + blockSize.y - 1) / blockSize.y);   //HOW MANY JUMPS BLOCKS WILL MAKE IN THE ENTIRE IMAGE

    black_white<<<gridSize, blockSize>>>(d_out, d_in, h_img.rows, h_img.cols, h_img.channels());
    cudaDeviceSynchronize();                    // waits for GPU

    cudaMemcpy(bw_img.data, d_out, IMG_SIZE_BYTES / h_img.channels(), cudaMemcpyDeviceToHost);

    cv::imwrite("../ggwp.jpg", bw_img);

    std::cout<<"B&W image ready!"<<std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}