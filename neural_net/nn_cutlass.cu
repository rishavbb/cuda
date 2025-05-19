#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
// #include <cmath>


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

    // printf("%f    %f    %f    \n", d_b1[0], d_b1[1], d_b1[2]);
    // printf("%f\n", d_b2[0]);
    updateW2B(d_w2, d_b2, d_y, d_hidden, d_out, d_a_out, hidden_size, lr);
    updateW1B(d_b1, d_w1, d_w2, d_y, d_out, d_a_out, d_a_hidden, d_hidden, d_x, input_size, hidden_size, lr);
    // printf("%f  %f\n", d_out[0], d_y[0]);

}

__global__ void sigmoid_kernel(float *data1, float *data2) {
    int idx = threadIdx.x;

    data2[idx] = 1.0f / (1.0f + expf(-1.0f * data1[idx]));
    
  }

int main(int argc, char *argv[]){

    float h_w1[INPUT_SIZE * HIDDEN_SIZE], h_w2[HIDDEN_SIZE * OUTPUT_SIZE];
    // float h_b1[HIDDEN_SIZE],              h_b2[OUTPUT_SIZE];
    float h_out[OUTPUT_SIZE];

    // Training data for XOR
    int dataset_size = 4;
    float h_x[dataset_size * INPUT_SIZE]  = {0,0,   0,1,    1,0,    1,1};
    float h_y[dataset_size * OUTPUT_SIZE] = { 0,     1,      1,      0 };

    random_initialize(h_w1, INPUT_SIZE * HIDDEN_SIZE);
    random_initialize(h_w2, HIDDEN_SIZE * OUTPUT_SIZE);


    float *d_x, *d_y;
    float *d_w1, *d_w2, *d_b1, *d_b2, *d_hidden, *d_a_hidden, *d_out,  *d_a_out;

    CHECK_CUDA(cudaMalloc(&d_x, dataset_size * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, dataset_size * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_a_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_a_hidden, 0, HIDDEN_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_out, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_out, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_a_out, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_a_out, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_w1, h_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w2, h_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_b1, 0, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b2, 0, OUTPUT_SIZE * sizeof(float)));

    // --------------------- THE BELOW COMMENTED CODE CAN BE USED FOR MATRIX MUL + SIGMOID. SINCE WE NEED VALUES PRE SIGMOID FOR BACKPROP, I AM NOT USING IT---------------------
    
    // using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    // using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 8>;
    // using InstructionShape = cutlass::gemm::GemmShape< 1,    1, 1>;


    // using EpilogueSigmoid = cutlass::epilogue::thread::LinearCombinationSigmoid<
    //     float,   // write float out to C
    //     1,       // one element per access
    //     float,   // accumulator type
    //     float    // alpha/beta compute type
    //     >;

    // using GemmWithSigmoid = cutlass::gemm::device::Gemm<
    //             float,                     // Element A
    //             cutlass::layout::RowMajor, // Layout A
    //             float,                     // Element B
    //             cutlass::layout::RowMajor, // Layout B
    //             float,                     // Element C / output
    //             cutlass::layout::RowMajor, // Layout C
    //             float,                     // ElementAccumulator
    //             cutlass::arch::OpClassSimt,// Operator class
    //             cutlass::arch::Sm70,       // SM architecture
    //             ThreadblockShape,
    //             WarpShape,
    //             InstructionShape,
    //             EpilogueSigmoid            // our custom epilogue
    //         >;

    // GemmWithSigmoid hidden_matmul, output_matmul;

    // --------------------- THE ABOVE COMMENTED CODE CAN BE USED FOR MATRIX MUL + SIGMOID. SINCE WE NEED VALUES PRE SIGMOID FOR BACKPROP, I AM NOT USING IT---------------------



    using Gemm = cutlass::gemm::device::Gemm<
                        float,                          // Element A
                        cutlass::layout::RowMajor,      // Layout A
                        float,                          // Element B
                        cutlass::layout::RowMajor,      // Layout B
                        float,                          // Element Output
                        cutlass::layout::RowMajor      // Layout Output
                 >;
    Gemm hidden_matmul, output_matmul;


    float alpha = 1.0f, beta = 1.0f;

    cutlass::Status status;

    for (int epoch=0; epoch<EPOCHS; epoch++){

        for (int prev_x=0, prev_y=0; prev_y<4; prev_x+=INPUT_SIZE, prev_y++){
        
            CHECK_CUDA(cudaMemcpy(d_x, &h_x[prev_x], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[prev_y], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            
            // d_x is [1x2] → M=1, K=2, row-major stride = number of columns =2.

            // d_w1 is [2x3] → K=2, N=3, row-major stride = number of columns =3.

            // d_b1 (the bias) is [1x3]; copying it into your output buffer so that your initial "d_a_hidden" holds B. That buffer has stride =3. 

            // the main formula for below is C = αAB + βC    (where A=d_x, B=d_w1, C=d_a_hidden (which is initially holding d_b1 value))
            CHECK_CUDA(cudaMemcpy(d_a_hidden, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

            status = hidden_matmul({
                { 1, 3, 2 },            // <M, N, K>
                { d_x, 2 },           // <A,  lda = K = 2>
                { d_w1, 3 },           // <B,  ldb = N = 3>
                { d_a_hidden, 3 },         // <d_a_hidden, ldc = 3>  contains the bias d_b1
                { d_a_hidden, 3 },         // <d_a_hidden, ldc = 3>
                { alpha, beta }         // <α=1, β=1>
              });
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Hidden GEMM failed: "
                            << cutlassGetStatusString(status) << "\n";
                std::exit(-1);
            }
            cudaDeviceSynchronize();

            sigmoid_kernel<<<1, HIDDEN_SIZE>>>(d_a_hidden, d_hidden);
            cudaDeviceSynchronize();
 
            //  d_hidden  [1x3]   M=1  K=3
            //  d_w2      [3x1]   K=3  N=1
            //  d_a_out   [1x1]
            CHECK_CUDA(cudaMemcpy(d_a_out, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

            status = output_matmul({
                { 1, 1, 3 },            // <M, N, K>
                { d_hidden, 3 },           // <A,  lda = K = 3>
                { d_w2, 1 },           // <B,  ldb = N = 1>
                { d_a_out, 1 },         // <d_a_out, ldc = 3>   contains the bias d_b2
                { d_a_out, 1 },         // <C_dst, ldc = 3>
                { alpha, beta }         // <α=1, β=1>
              });
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Output GEMM failed: "
                            << cutlassGetStatusString(status) << "\n";
                std::exit(-1);
            }
            cudaDeviceSynchronize();


            sigmoid_kernel<<<1, OUTPUT_SIZE>>>(d_a_out, d_out);
            cudaDeviceSynchronize();
            
            // printf("GGWP2 \n");
            backwardPass<<<1, INPUT_SIZE*HIDDEN_SIZE>>>(d_w1, d_w2, d_b1, d_b2, d_y, d_hidden, d_a_hidden, d_out, d_a_out, d_x, INPUT_SIZE, HIDDEN_SIZE, LR);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

        }
    }

    for (int prev_x=0, prev_y=0; prev_y<4; prev_x+=INPUT_SIZE, prev_y++){
        CHECK_CUDA(cudaMemcpy(d_x, &h_x[prev_x], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, &h_y[prev_y], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaMemcpy(d_a_hidden, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

        status = hidden_matmul({
            { 1, 3, 2 },            // <M, N, K>
            { d_x, 2 },           // <A,  lda = K = 2>
            { d_w1, 3 },           // <B,  ldb = N = 3>
            { d_a_hidden, 3 },         // <d_a_hidden, ldc = 3>  contains the bias d_b1
            { d_a_hidden, 3 },         // <d_a_hidden, ldc = 3>
            { alpha, beta }         // <α=1, β=1>
          });
        cudaDeviceSynchronize();


        sigmoid_kernel<<<1, HIDDEN_SIZE>>>(d_a_hidden, d_hidden);
        cudaDeviceSynchronize();

        //  d_hidden  [1x3]   M=1  K=3
        //  d_w2      [3x1]   K=3  N=1
        //  d_a_out   [1x1]
        CHECK_CUDA(cudaMemcpy(d_a_out, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
        

        status = output_matmul({
            { 1, 1, 3 },            // <M, N, K>
            { d_hidden, 3 },           // <A,  lda = K = 3>
            { d_w2, 1 },           // <B,  ldb = N = 1>
            { d_a_out, 1 },         // <d_a_out, ldc = 3>  contains the bias d_b1
            { d_a_out, 1 },         // <C_dst, ldc = 3>
            { alpha, beta }         // <α=1, β=1>
          });
        cudaDeviceSynchronize();

        sigmoid_kernel<<<1, OUTPUT_SIZE>>>(d_a_out, d_out);
        cudaDeviceSynchronize();

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