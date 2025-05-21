#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>




#define CHECK_CUDA(call){               \
    cudaError_t err = call;             \
    if (err != cudaSuccess){            \
        std::cerr<<"CUDA Error in file "<<__FILE__<<" Line number:"<<__LINE__<< "  Error: "<< cudaGetErrorString(err) <<std::endl;  \
        exit(EXIT_FAILURE);             \
    }                                   \
}                                       \

#define CHECK_CUDNN(call)                                                         \
  do {                                                                             \
    cudnnStatus_t status = (call);                                                \
    if (status != CUDNN_STATUS_SUCCESS) {                                          \
      std::cerr                                                                \
        << "cuDNN Error in file " << __FILE__                                    \
        << " at line " << __LINE__                                               \
        << ": " << cudnnGetErrorString(status)                                   \
        << std::endl;                                                             \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                              \
  } while (0)

#define CHECK_CUBLAS(call)                                                    \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr                                                             \
        << "cuBLAS Error in " << __FILE__                                     \
        << " at line " << __LINE__                                            \
        << ": " << status                                                    \
        << std::endl;                                                         \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                          \
  } while (0)



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

__global__ void mse_backward_kernel(float *Y, float *Y_true, float *dY, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dY[i] = 2.0f * (Y[i] - Y_true[i]) / float(N);
    }
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
    float *d_w1, *d_w2, *d_b1, *d_b2, *d_hidden, *d_a_hidden, *d_out, *d_a_out, *d_y_delta, *d_B_delta;


    float *d_dw1, *d_dw2, *d_db1, *d_db2, *d_dh;
    CHECK_CUDA(cudaMalloc(&d_dw1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dw2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db2, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dh, HIDDEN_SIZE * sizeof(float)));


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

    CHECK_CUDA(cudaMalloc(&d_y_delta, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_y_delta, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_a_out, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_a_out, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_w1, h_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w2, h_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_b1, 0, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b2, 0, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_B_delta, OUTPUT_SIZE*sizeof(float)));


    // Handles
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    CHECK_CUBLAS(cublasCreate(&cublas));


    // Descriptors for Layer 1: Input -> Hidden
    cudnnTensorDescriptor_t xDesc, hDesc, b1Desc;
    cudnnFilterDescriptor_t w1Desc;
    cudnnConvolutionDescriptor_t conv1Desc;
    cudnnActivationDescriptor_t act1Desc;

    // Input tensor: N=1, C=INPUT_SIZE, H=W=1
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                    xDesc,              //tensDesc
                    CUDNN_TENSOR_NCHW,  // format
                    CUDNN_DATA_FLOAT,   // datatype
                    1,                  // batch
                    INPUT_SIZE,         // channels = input features
                    1, 1));             // H, W


    // Filter (weights): K=HIDDEN_SIZE, C=INPUT_SIZE, H=W=1
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w1Desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
                    w1Desc,
                    CUDNN_DATA_FLOAT,   // datatype
                    CUDNN_TENSOR_NCHW,  // format
                    HIDDEN_SIZE,        // K
                    INPUT_SIZE,         // C
                    1, 1));             // H, W


    // Bias: N=1, C=HIDDEN_SIZE, H=W=1
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&b1Desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                    b1Desc,             //tensDesc
                    CUDNN_TENSOR_NCHW,  // format
                    CUDNN_DATA_FLOAT,   // datatype
                    1,                  // batch
                    HIDDEN_SIZE,        // channels = HIDDEN_SIZE
                    1, 1));             // H, W


    // Convolution descriptor: pad=0, stride=1, dilation=1
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
                    conv1Desc,
                    0,0,                // pad_h, pad_w
                    1,1,                // stride_h, stride_w
                    1,1,                // dilation_h, dilation_w
                    CUDNN_CROSS_CORRELATION,
                    CUDNN_DATA_FLOAT));


    // Activation (sigmoid)
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&act1Desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
                    act1Desc,
                    CUDNN_ACTIVATION_SIGMOID,
                    CUDNN_PROPAGATE_NAN,
                    0.0));
    


    // Output (hidden) tensor: N=1, C=HIDDEN_SIZE, H=W=1
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        hDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, HIDDEN_SIZE, 1, 1));


    // Descriptors for Layer 2: Hidden -> Output
    cudnnTensorDescriptor_t hOutDesc, yDesc, b2Desc, d_y_deltaDesc, d_b1_deltaDesc, d_b2_deltaDesc;
    cudnnFilterDescriptor_t w2Desc;
    cudnnConvolutionDescriptor_t conv2Desc;
    cudnnActivationDescriptor_t act2Desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&d_y_deltaDesc));
    CHECK_CUDNN( cudnnSetTensor4dDescriptor(
        d_y_deltaDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,            // N = batch size
        OUTPUT_SIZE,  // C = number of biases
        1,            // H
        1             // W
    ) );

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&d_b1_deltaDesc));
    CHECK_CUDNN( cudnnSetTensor4dDescriptor(
        d_b1_deltaDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,            // N = batch size
        OUTPUT_SIZE,  // C = number of biases
        1,            // H
        1             // W
    ) );

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&d_b2_deltaDesc));
    CHECK_CUDNN( cudnnSetTensor4dDescriptor(
        d_b2_deltaDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,            // N = batch size
        OUTPUT_SIZE,  // C = number of biases
        1,            // H
        1             // W
    ) );
    
    // Hidden input: N=1, C=HIDDEN_SIZE
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hOutDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                    hOutDesc,           // tensDesc
                    CUDNN_TENSOR_NCHW,  // format
                    CUDNN_DATA_FLOAT,   // datatype
                    1,                  // batch
                    HIDDEN_SIZE,        // channels = HIDDEN_SIZE
                    1, 1));             // H, W


    // Filter (weights): K=OUTPUT_SIZE, C=HIDDEN_SIZE
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w2Desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
                    w2Desc,
                    CUDNN_DATA_FLOAT,   // datatype
                    CUDNN_TENSOR_NCHW,  // format
                    OUTPUT_SIZE,        // K
                    HIDDEN_SIZE,        // C
                    1, 1));             // H, W

    // Bias: N=1, C=OUTPUT_SIZE
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&b2Desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                    b2Desc,             // tensDesc
                    CUDNN_TENSOR_NCHW,  // format
                    CUDNN_DATA_FLOAT,   // datatype
                    1,                  // batch
                    OUTPUT_SIZE,        // channels = OUTPUT_SIZE
                    1, 1));             // H, W


    // Convolution descriptor
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
                    conv2Desc,
                    0,0,                // pad_h, pad_w
                    1,1,                // stride_h, stride_w
                    1,1,                // dilation_h, dilation_w
                    CUDNN_CROSS_CORRELATION,
                    CUDNN_DATA_FLOAT));

    // Activation (sigmoid)
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&act2Desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
                    act2Desc,
                    CUDNN_ACTIVATION_SIGMOID,
                    CUDNN_PROPAGATE_NAN,
                    0.0));

    // Output tensor: N=1, C=OUTPUT_SIZE
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                    yDesc,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    1,
                    OUTPUT_SIZE,
                    1, 1));


    //  -------------------------------- FORWARD ALGO --------------------------------
    cudnnConvolutionFwdAlgo_t fwdAlgo1, fwdAlgo2;

    cudnnConvolutionFwdAlgoPerf_t perfFwd1[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int returnedFwdCount1 = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                    cudnn,          // handle
                    xDesc,          // xDesc    : cudnnTensorDescriptor_t
                    w1Desc,          // wDesc    : cudnnFilterDescriptor_t
                    conv1Desc,      // convDesc : cudnnConvolutionDescriptor_t
                    hDesc,          // yDesc    : cudnnTensorDescriptor_t
                    CUDNN_CONVOLUTION_FWD_ALGO_COUNT,         // “how many results do you want?”
                    &returnedFwdCount1,
                    perfFwd1));
    fwdAlgo1 = perfFwd1[0].algo;


    cudnnConvolutionFwdAlgoPerf_t perfFwd2[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int returnedFwdCount2 = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                    cudnn,          // handle
                    hDesc,          // xDesc    : cudnnTensorDescriptor_t
                    w2Desc,          // wDesc    : cudnnFilterDescriptor_t
                    conv2Desc,      // convDesc : cudnnConvolutionDescriptor_t
                    yDesc,          // yDesc    : cudnnTensorDescriptor_t
                    CUDNN_CONVOLUTION_FWD_ALGO_COUNT,         // “how many results do you want?”
                    &returnedFwdCount2,
                    perfFwd2));
    fwdAlgo2 = perfFwd2[0].algo;



    // This function returns the amount of GPU memory workspace the user needs to allocate
    // to be able to call cudnnConvolutionForward() with the specified algorithm. 
    size_t fwdWorkspaceSize1 = 0, fwdWorkspaceSize2 = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnn,
                    xDesc,
                    w1Desc,
                    conv1Desc,
                    hDesc,
                    fwdAlgo1,
                    &fwdWorkspaceSize1));

    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnn,
                    hDesc,
                    w2Desc,
                    conv2Desc,
                    yDesc,
                    fwdAlgo2,
                    &fwdWorkspaceSize2));
    //  -------------------------------- FORWARD ALGO --------------------------------




    //  -------------------------------- BACKWARD ALGO --------------------------------
    // Backward filter algo

    cudnnConvolutionBwdFilterAlgoPerf_t perfBwdFilter1[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT],
                                        perfBwdFilter2[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    int returnedBwdFilterCount1 = 0, returnedBwdFilterCount2 = 0;


    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo1, bwdFilterAlgo2;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                    cudnn,
                    xDesc,
                    hDesc,
                    conv1Desc,
                    w1Desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                    &returnedBwdFilterCount1,
                    perfBwdFilter1));
    bwdFilterAlgo1 = perfBwdFilter1[0].algo;

    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                    cudnn,
                    hDesc,
                    yDesc,
                    conv2Desc,
                    w2Desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                    &returnedBwdFilterCount2,
                    perfBwdFilter2));
    bwdFilterAlgo2 = perfBwdFilter2[0].algo;

    size_t bwdFilterWksz1 = 0, bwdFilterWksz2 = 0;

    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnn,
                    xDesc,
                    hDesc,
                    conv1Desc,
                    w1Desc,
                    bwdFilterAlgo1,
                    &bwdFilterWksz1));

    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnn,
                    hDesc,
                    yDesc,
                    conv2Desc,
                    w2Desc,
                    bwdFilterAlgo2,
                    &bwdFilterWksz2));
                    
                    
    
    // Backward‑data algo
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo1, bwdDataAlgo2;
    cudnnConvolutionBwdDataAlgoPerf_t perfBwdData1[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT], perfBwdData2[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
    int returnedBwdDataCount1 = 0, returnedBwdDataCount2 = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    cudnn,
                    w1Desc,
                    hDesc,
                    conv1Desc,
                    xDesc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                    &returnedBwdDataCount1,
                    perfBwdData1));

    bwdDataAlgo1 = perfBwdData1[0].algo;

    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    cudnn,
                    w2Desc,
                    yDesc,
                    conv2Desc,
                    hDesc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                    &returnedBwdDataCount2,
                    perfBwdData2));
    bwdDataAlgo2 = perfBwdData2[0].algo;


    size_t bwdDataWksz1 = 0, bwdDataWksz2 = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn,
                    w1Desc,
                    hDesc,
                    conv1Desc,
                    xDesc,
                    bwdDataAlgo1,
                    &bwdDataWksz1));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn,
                    w2Desc,
                    yDesc,
                    conv2Desc,
                    hDesc,
                    bwdDataAlgo2,
                    &bwdDataWksz2));
    //  -------------------------------- BACKWARD ALGO --------------------------------


    // size_t workspaceSize*;

    void *workspace;
    int maxWksz = std::max({fwdWorkspaceSize1, fwdWorkspaceSize2, bwdFilterWksz1, bwdFilterWksz2, bwdDataWksz1, bwdDataWksz2});
    cudaMalloc(&workspace, maxWksz);



    float alpha = 1.0f, beta = 0.0f;
    int batch_size = 1;
    int N = batch_size * OUTPUT_SIZE;
    int threads = 32, blocks = (N + threads - 1)/threads;
    float loss;

    for (int epoch=0; epoch<EPOCHS; epoch++){
        loss = 0.0f;
        
        for (int sample=0; sample<dataset_size; sample++){
            int prev_x = sample * INPUT_SIZE;
            int prev_y = sample;
        
            // Copy the current sample to device
            CHECK_CUDA(cudaMemcpy(d_x, &h_x[prev_x], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[prev_y], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
            // Forward pass: Input to Hidden Layer
            // FORMULA y = act (alpha1 * conv(x) + alpha2 * z + bias)
            CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
                            cudnn,              // handle
                            &alpha,             // alpha1
                            xDesc,              // xDesc
                            d_x,                // x
                            w1Desc,             // wDesc
                            d_w1,               // w
                            conv1Desc,          // convDesc,
                            fwdAlgo1,           // algo
                            workspace,          // workSpace
                            fwdWorkspaceSize1,  // workSpaceSizeInBytes
                            &beta,              // alpha2
                            hDesc,              // zDesc
                            d_a_hidden,         // z (pre-activation)
                            b1Desc,             // bDesc   
                            d_b1,               // bias
                            act1Desc,           // actDesc
                            hDesc,              // yDesc
                            d_hidden));         // y (post-activation)
    
            // Forward pass: Hidden to Output Layer
            CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
                            cudnn,              // handle
                            &alpha,             // alpha1
                            hDesc,              // xDesc
                            d_hidden,           // x
                            w2Desc,             // wDesc
                            d_w2,               // w
                            conv2Desc,          // convDesc,
                            fwdAlgo2,           // algo
                            workspace,          // workSpace
                            fwdWorkspaceSize2,  // workSpaceSizeInBytes
                            &beta,              // alpha2
                            yDesc,              // zDesc
                            d_a_out,            // z (pre-activation)
                            b2Desc,             // bDesc   
                            d_b2,               // bias
                            act2Desc,           // actDesc
                            yDesc,              // yDesc
                            d_out));            // y (post-activation)
            
            // Calculate loss (MSE) and gradients for output layer
            mse_backward_kernel<<<blocks,threads>>>(d_out, d_y, d_y_delta, N);
            cudaDeviceSynchronize();
    
            // Backward pass
            
            // 1. Backward for output layer bias
            CHECK_CUDNN(cudnnConvolutionBackwardBias(
                            cudnn,
                            &alpha,
                            d_y_deltaDesc,
                            d_y_delta,
                            &beta,
                            b2Desc,
                            d_db2));
            
            // 2. Backward for output layer weights
            CHECK_CUDNN(cudnnConvolutionBackwardFilter(
                            cudnn,
                            &alpha,
                            hDesc,              // x descriptor
                            d_hidden,           // x (hidden layer activations)
                            d_y_deltaDesc,      // dy descriptor
                            d_y_delta,          // dy (output gradient)
                            conv2Desc,          // convolution descriptor
                            bwdFilterAlgo2,     // algorithm
                            workspace,          // workspace
                            bwdFilterWksz2,     // workspace size
                            &beta,              // beta
                            w2Desc,             // dw descriptor 
                            d_dw2));            // dw (weight gradient)
            
            // 3. Backward for hidden layer (propagate error)
            CHECK_CUDNN(cudnnConvolutionBackwardData(
                            cudnn,
                            &alpha,
                            w2Desc,             // w descriptor
                            d_w2,               // w (weights)
                            d_y_deltaDesc,      // dy descriptor
                            d_y_delta,          // dy (output gradient)
                            conv2Desc,          // convolution descriptor
                            bwdDataAlgo2,       // algorithm
                            workspace,          // workspace
                            bwdDataWksz2,       // workspace size
                            &beta,              // beta
                            hDesc,              // dx descriptor
                            d_dh));             // dx (hidden gradient)
                            
            // 4. Apply activation backward for hidden layer gradient
            CHECK_CUDNN(cudnnActivationBackward(
                            cudnn,
                            act1Desc,           // activation descriptor
                            &alpha,             // alpha
                            hDesc,              // y descriptor
                            d_hidden,           // y (activation output)
                            hDesc,              // dy descriptor
                            d_dh,               // dy (gradient from next layer)
                            hDesc,              // x descriptor
                            d_a_hidden,         // x (activation input)
                            &beta,              // beta
                            hDesc,              // dx descriptor
                            d_dh));             // dx (input gradient)
            
            // 5. Backward for hidden layer bias
            CHECK_CUDNN(cudnnConvolutionBackwardBias(
                            cudnn,
                            &alpha,
                            hDesc,              // dy descriptor
                            d_dh,               // dy (hidden gradient)
                            &beta,              // beta
                            b1Desc,             // db descriptor
                            d_db1));            // db (bias gradient)
            
            // 6. Backward for input-to-hidden weights
            CHECK_CUDNN(cudnnConvolutionBackwardFilter(
                            cudnn,
                            &alpha,             // alpha
                            xDesc,              // x descriptor
                            d_x,                // x (input)
                            hDesc,              // dy descriptor
                            d_dh,               // dy (hidden gradient)
                            conv1Desc,          // convolution descriptor
                            bwdFilterAlgo1,     // algorithm
                            workspace,          // workspace
                            bwdFilterWksz1,     // workspace size
                            &beta,              // beta
                            w1Desc,             // dw descriptor
                            d_dw1));            // dw (weight gradient)
            
            // Update weights and biases using SGD
            float learning_rate = -LR;  // Negative because we're performing gradient descent
            
            // Update weights using cublas, FORMULA y := α * x + y
            CHECK_CUBLAS(cublasSaxpy(cublas,                    // handle
                                     INPUT_SIZE * HIDDEN_SIZE,  // length of x and y
                                    &learning_rate,             // α
                                    d_dw1,                      // x: gradient vector
                                    1,                          // stride in x
                                    d_w1,                       // y: weight vector to update
                                    1));                        // // stride in y
            
            CHECK_CUBLAS(cublasSaxpy(cublas, HIDDEN_SIZE * OUTPUT_SIZE, 
                                    &learning_rate, d_dw2, 1, d_w2, 1));
            
            // Update biases
            CHECK_CUBLAS(cublasSaxpy(cublas, HIDDEN_SIZE, 
                                    &learning_rate, d_db1, 1, d_b1, 1));
            
            CHECK_CUBLAS(cublasSaxpy(cublas, OUTPUT_SIZE, 
                                    &learning_rate, d_db2, 1, d_b2, 1));
            
            // Calculate and accumulate loss for monitoring
            float h_pred[OUTPUT_SIZE], h_true[OUTPUT_SIZE];
            CHECK_CUDA(cudaMemcpy(h_pred, d_out, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_true, d_y, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            
            loss += (h_pred[0] - h_true[0]) * (h_pred[0] - h_true[0]);
        }
        
        // Print loss every 1000 epochs
        if (epoch % 1000 == 0) {
            loss /= dataset_size;
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
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

    cudaFree(d_dw1);
    cudaFree(d_dw2);
    cudaFree(d_db1);
    cudaFree(d_db2);
    cudaFree(d_dh);
    cudaFree(workspace);

    
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(hDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(b1Desc);
    cudnnDestroyTensorDescriptor(b2Desc);
    cudnnDestroyTensorDescriptor(d_y_deltaDesc);
    cudnnDestroyTensorDescriptor(d_b1_deltaDesc);
    cudnnDestroyTensorDescriptor(d_b2_deltaDesc);
    cudnnDestroyFilterDescriptor(w1Desc);
    cudnnDestroyFilterDescriptor(w2Desc);
    cudnnDestroyConvolutionDescriptor(conv1Desc);
    cudnnDestroyConvolutionDescriptor(conv2Desc);
    cudnnDestroyActivationDescriptor(act1Desc);
    cudnnDestroyActivationDescriptor(act2Desc);

    cudnnDestroy(cudnn);
    cublasDestroy(cublas);

}