# 🧠 CUDA Neural Network Playground

Welcome to the **CUDA Neural Network Playground**! This project showcases a minimalist neural network implemented in three powerful CUDA-based styles:

* 🚀 **Raw CUDA** (from scratch)
* 🛠️ **CUTLASS** (NVIDIA's CUDA C++ Templates)
* ⚙️ **cuDNN** (NVIDIA's Deep Neural Network library)

Whether you're benchmarking performance, exploring low-level GPU programming, or learning how libraries abstract complexity, this repo is a practical playground for neural net enthusiasts.

---

## 📁 Project Structure

```
├── nn_cuda.cu              # Raw CUDA implementation
├── nn_cutlass.cu           # CUTLASS-accelerated version
├── nn_cudnn.cu             # cuDNN-accelerated version
├── README.md               # You're here :)
```

---

## 🔧 Setup Instructions

### 1. Prerequisites

* CUDA Toolkit 11.x or later
* NVIDIA GPU with Compute Capability 7.0+
* [CUTLASS Library](https://github.com/NVIDIA/cutlass)
* cuDNN Library (bundled with CUDA or install separately)

### 2. Build & Run Commands

#### ⚡ Raw CUDA

```bash
nvcc nn_cuda.cu -o nn_cuda
./nn_cuda
```

#### 🛠️ CUTLASS

```bash
git clone https://github.com/NVIDIA/cutlass.git ~/third_party/cutlass

nvcc -std=c++17 \
  -Xcompiler -std=c++17 \
  -I ~/third_party/cutlass/include \
  -I ~/third_party/cutlass/tools/util/include \
  nn_cutlass.cu -o nn_cutlass --expt-relaxed-constexpr

./nn_cutlass
```

#### ⚙️ cuDNN

```bash
nvcc nn_cudnn.cu -lcudnn -lcublas -gencode arch=compute_86,code=sm_86 -o nn_cudnn
./nn_cudnn
```

---

## 🚧 What It Does

This project implements a **basic feedforward neural network** with:

* One hidden layer
* Sigmoid activation
* Mean Squared Error loss
* Backpropagation (in CUDA)

The same logic is ported to CUTLASS and cuDNN to demonstrate abstraction vs. control vs. performance.
