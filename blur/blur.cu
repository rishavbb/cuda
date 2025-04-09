#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <iostream>


__global__ void blur(uchar *d_in, uchar *d_out, int rows, int cols){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y>=rows){
        return;
    }

    int arr[9][2] = {{-1,-1}, {-1,0}, {-1,1},
                     {0,-1}, {0, 0}, {0,1},
                     {1,-1}, {1,0}, {1,1}};

    uchar R=0, G=0, B=0;
    
    int x_new, y_new, rgbIdx;

    for (int i=0; i<9;i++){
        x_new = x + arr[i][1];
        y_new = y + arr[i][0];

        if (x_new>=cols || x_new<0 || y_new>=rows || y_new<0)
            continue;

        rgbIdx = (y_new*cols + x_new)*3;

        R += d_in[rgbIdx] * 0.1;
        G += d_in[rgbIdx+1] * 0.1;
        B += d_in[rgbIdx+2] * 0.1;
    }

    int pixel_idx = (y*cols + x) *3;
    d_out[pixel_idx] = R;
    d_out[pixel_idx+1] = G;
    d_out[pixel_idx+2] = B;

}
int main(int argc, char** argv){
    cv::Mat h_in = cv::imread("../cinque_terre_small.jpg");

    int IMG_SIZE_BYTES = h_in.rows * h_in.cols * h_in.elemSize();

    uchar *d_in, *d_out;

    cv::Mat blur_img(h_in.rows, h_in.cols, CV_8UC3);
    cudaMalloc((void**)&d_in, IMG_SIZE_BYTES);
    cudaMalloc((void**)&d_out, IMG_SIZE_BYTES);

    cudaMemcpy(d_in, h_in.data, IMG_SIZE_BYTES, cudaMemcpyHostToDevice);
    

    dim3 blockSize(16,16);
    dim3 gridSize((h_in.cols + blockSize.x - 1) / blockSize.x,
                  (h_in.rows + blockSize.y - 1) / blockSize.y);

    blur<<<gridSize, blockSize>>>(d_in, d_out, h_in.rows, h_in.cols);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaMemcpy(blur_img.data, d_out, IMG_SIZE_BYTES, cudaMemcpyDeviceToHost);

    cv::imwrite("../blur.jpg", blur_img);

    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}