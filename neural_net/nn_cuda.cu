#include <cuda_runtime.h>
#include <random>
#include <iostream>

#define CHECK_CUDA(call){               \
    cudaError_t err = call;             \
    if (err != cudaSuccess){            \
        std::cerr<<"CUDA Error in file "<<__FILE__<<" Line number:"<<__LINE__<< "  Error: "<< cudaGetErrorString(err) <<std::endl;  \
        exit(EXIT_FAILURE);             \
    }                                   \
}                                       \


int INPUT_SIZE = 2, HIDDEN_SIZE = 3, OUTPUT_SIZE = 1;
float LR = 0.1f;
int EPOCHS = 10000;


// Choose seed as 42
std::mt19937 gen(42);

// Define the distribution range [-1, +1]
std::uniform_real_distribution<float> dist(-1.0, 1.0);


void random_initialize(float *arr, int size){

    for(int i=0; i<size; i++){
        arr[i] = dist(gen);
    }
}

__device__ void computeHidden(float *d_x, float *d_w1, float *d_b1, float *d_hidden, float *d_a_hidden, int input_size, int hidden_size){

    int idx = threadIdx.x;

    // printf("%d   %f  %f  %f  %f\n", blockIdx.x, d_x[0], d_x[1], d_x[2], d_x[3]);
    if (idx < hidden_size){
        
        for (int i=0; i<input_size;i++){
            d_a_hidden[idx] = d_a_hidden[idx] + d_x[i] * d_w1[idx * input_size + i];    //                   TODO    try += 
            // printf("%f  %f   %f  \n", d_a_hidden[idx], d_x[blockIdx.x * input_size + i], d_w1[idx*input_size + i]);
            
        }
        d_a_hidden[idx] += d_b1[idx];


        d_hidden[idx] = 1.0f / (1.0f + expf(-1.0f * d_a_hidden[idx]));        // Activation
    }
    __syncthreads();
}

__device__ void computeOutput(float *d_out, float *d_a_out, float *d_w2, float *d_b2, float *d_hidden, int output_size, int hidden_size){
    int idx = threadIdx.x;

    if (idx<output_size){

        for (int i=0; i<hidden_size; i++){
            d_a_out[idx] =  d_a_out[idx] + d_hidden[i] * d_w2[idx*hidden_size + i];
        }
        d_a_out[idx] += d_b2[idx];


        d_out[idx] = 1.0f / (1.0f + expf(-1.0f * d_a_out[idx]));        // Activation
    
    }
    __syncthreads();
    
}

__global__ void forwardPass(float *d_x, float *d_w1, float *d_w2, float *d_b1, 
                            float *d_b2, float *d_hidden, float *d_a_hidden, 
                            float *d_out, float *d_a_out, int input_size,
                            int output_size, int hidden_size){
    
    // printf("%f  %f  %f  %f\n", d_x[0], d_x[1], d_x[2], d_x[3]);
    computeHidden(d_x, d_w1, d_b1, d_hidden, d_a_hidden, input_size, hidden_size);
    computeOutput(d_out, d_a_out, d_w2, d_b2, d_hidden, output_size, hidden_size);
    
}

__device__ void updateW2B(float *d_w2, float *d_b2, float *d_y,
                          float *d_hidden, float *d_out, float *d_a_out,
                          int hidden_size, float lr){

    int idx = threadIdx.x;

    if (idx < hidden_size){

        float delta = (-2.0) * (d_y[0] - d_out[0]) * (d_out[0] * (1.0 - d_out[0]));
        
        if (idx==0){
            d_b2[idx] -= lr * delta;
        }
        d_w2[idx] -= lr * delta * d_hidden[idx];

    }
    __syncthreads();

}

__device__ void updateW1B(float* d_b1, float *d_w1, float *d_w2, float *d_y,
                          float *d_out, float *d_a_out, float *d_a_hidden,
                          float *d_hidden, float *d_x, int input_size,
                          int hidden_size, float lr){

    int idx = threadIdx.x;

    if (idx < input_size*hidden_size){
        float delta = (-2.0) * (d_y[0] - d_out[0]) * (d_out[0] * (1.0 - d_out[0])) * d_w2[idx/2] * (d_hidden[idx/2] * (1 - d_hidden[idx/2]));
        
        if (idx < hidden_size){
            d_b1[idx] -= lr * delta;
        }
        d_w1[idx] -= lr * delta * d_x[idx%input_size];

    }
    __syncthreads();

}


__global__ void backwardPass(float *d_w1, float *d_w2, float *d_b1, float *d_b2,
                             float *d_y, float *d_hidden, float *d_a_hidden, float *d_out,
                             float *d_a_out, float *d_x, int input_size, int hidden_size,
                             float lr){

    updateW2B(d_w2, d_b2, d_y, d_hidden, d_out, d_a_out, hidden_size, lr);
    updateW1B(d_b1, d_w1, d_w2, d_y, d_out, d_a_out, d_a_hidden, d_hidden, d_x, input_size, hidden_size, lr);
    // printf("%f  %f\n", d_out[0], d_y[0]);

}


int main(int argc, char *argv[]){

    float h_w1[INPUT_SIZE * HIDDEN_SIZE], h_w2[HIDDEN_SIZE * OUTPUT_SIZE];
    float h_b1[HIDDEN_SIZE],              h_b2[OUTPUT_SIZE];
    float h_out[OUTPUT_SIZE];

    // Training data for XOR
    int dataset_size = 4;
    float h_x[dataset_size * INPUT_SIZE]  = {0,0,   0,1,    1,0,    1,1};
    float h_y[dataset_size * OUTPUT_SIZE] = { 0,     1,      1,      0 };

    random_initialize(h_w1, INPUT_SIZE * HIDDEN_SIZE);
    random_initialize(h_w2, HIDDEN_SIZE * OUTPUT_SIZE);
    // random_initialize(h_out);
    random_initialize(h_b1, HIDDEN_SIZE);
    random_initialize(h_b2, OUTPUT_SIZE);

    float *d_x, *d_y;
    float *d_w1, *d_w2, *d_b1, *d_b2, *d_hidden, *d_out, *d_a_hidden, *d_a_out;

    CHECK_CUDA(cudaMalloc(&d_x, dataset_size * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, dataset_size * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_a_hidden, HIDDEN_SIZE * sizeof(float)));
    // CHECK_CUDA(cudaMemset(d_a_hidden, 0, HIDDEN_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_out, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_out, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_a_out, OUTPUT_SIZE * sizeof(float)));
    // CHECK_CUDA(cudaMemset(d_a_out, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_w1, h_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w2, h_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));



    dim3 blockSize(HIDDEN_SIZE);                                                                // TODO MAX OF HIDDEN AND INPUT NUMBERS
    // dim3 gridSize((dataset_size + (batch_size-1)) / batch_size);

    for (int epoch=0; epoch<EPOCHS; epoch++){

        for (int prev_x=0, prev_y=0; prev_y<4; prev_x+=INPUT_SIZE, prev_y++){
        
            CHECK_CUDA(cudaMemcpy(d_x, &h_x[prev_x], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[prev_y], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            
            // CHECK_CUDA(cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_a_hidden, 0, HIDDEN_SIZE * sizeof(float)));
            // CHECK_CUDA(cudaMemset(d_out, 0, OUTPUT_SIZE * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_a_out, 0, OUTPUT_SIZE * sizeof(float)));

            forwardPass<<<1, blockSize>>>(d_x, d_w1, d_w2, d_b1, d_b2, d_hidden, d_a_hidden, d_out, d_a_out, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());   
            backwardPass<<<1, INPUT_SIZE*HIDDEN_SIZE>>>(d_w1, d_w2, d_b1, d_b2, d_y, d_hidden, d_a_hidden, d_out, d_a_out, d_x, INPUT_SIZE, HIDDEN_SIZE, LR);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            // printf("prev_x %d    prev_y %d\n", prev_x, prev_y);
        }
    }

    for (int prev_x=0, prev_y=0; prev_y<4; prev_x+=INPUT_SIZE, prev_y++){
        CHECK_CUDA(cudaMemcpy(d_x, &h_x[prev_x], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(d_y, &h_y[prev_y], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemset(d_a_hidden, 0, HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_a_out, 0, OUTPUT_SIZE * sizeof(float)));

        forwardPass<<<1, blockSize>>>(d_x, d_w1, d_w2, d_b1, d_b2, d_hidden, d_a_hidden, d_out, d_a_out, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out, d_out, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        printf("Original Answer:%f \t Predicted Answer:%f \n", h_y[prev_y], h_out[0]);

    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_hidden);
    cudaFree(d_a_hidden);
    cudaFree(d_out);
    cudaFree(d_a_out);

}