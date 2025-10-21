#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1: Each thread within a block calculates partial sum of its assigned elements in @inp_f4
  // Initialize per-thread accumulators to zero to avoid reading uninitialized memory
  // Use single-element arrays for l_sum_x and l_sum_x2 because blockReduce requires array arguments.
  float l_sum_x[1] = {0.0f};
  float l_sum_x2[1] = {0.0f};
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    // Accumulate partial sums for this thread
    l_sum_x[0] += val.x + val.y + val.z + val.w;
    l_sum_x2[0] += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2
  // Speedup can be achieved by computing the standard deviation as:
  // σ_x = √(E[x²] - E[x]² + ε)
  
  
  // reduce across the block: blockReduce only supports fixed template sizes
  // here we have a single value per thread (summing across lanes), so use 1
  blockReduce<ReduceType::kSum, 1>(l_sum_x);
  blockReduce<ReduceType::kSum, 1>(l_sum_x2);


  // Thread 0 finishes the math
  if (threadIdx.x == 0) {
    float mean = l_sum_x[0] / (hidden_size * 4);
    float mean2 = l_sum_x2[0] / (hidden_size * 4);
    float var = mean2 - mean * mean + LN_EPSILON;

    means[blockIdx.x] = mean;
    vars[blockIdx.x] = var;
  }

  __syncthreads();


  // Step 3 normalize and apply scale/bias, write outputs
  float4 *out_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f  = reinterpret_cast<const float4 *>(bias);

  // Load mean/var once per block (already written by thread 0) to avoid
  // repeated global reads inside the per-thread loop.
  float mean = means[blockIdx.x];
  float var = vars[blockIdx.x];
  float rstd = rsqrtf(var);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 tmp;
    tmp.x = (inp_f4[idx].x - mean) * rstd * scale_f[idx].x + bias_f[idx].x;
    tmp.y = (inp_f4[idx].y - mean) * rstd * scale_f[idx].y + bias_f[idx].y;
    tmp.z = (inp_f4[idx].z - mean) * rstd * scale_f[idx].z + bias_f[idx].z;
    tmp.w = (inp_f4[idx].w - mean) * rstd * scale_f[idx].w + bias_f[idx].w;
    out_f4[idx] = tmp;
  }
  
  
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  // Shared memory arrays betta_buffer and gamma_buffer are declared to store intermediate results within the thread block.
  
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM]; 
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  //CUDA thread blocks cg::thread_block and thread block tiles cg::thread_block_tile are used to organize threads.
  // Block is TILE_DIM × TILE_DIM threads.
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1 Compute the partial gradients by looping across inp rows.
  // Loop Over Rows: Threads in the y-dimension loop over rows,
  // calculating partial gradients for each row based on the
  // given inputs out_grad, inp, means, vars.

  // Use scalar accumulators (no float4)
  float partial_betta = 0.0f;
  float partial_gamma = 0.0f;

  // Compute column index in the global width
  int col = blockIdx.x * TILE_DIM + threadIdx.x;

  // Loop over rows (batch_size * seq_len). Threads stride by TILE_DIM in y.
  for (int row = threadIdx.y; row < rows; row += TILE_DIM) {
    if (col >= width) break; // this thread maps outside actual width

    int idx = row * width + col;

    // Load dout (out_grad)
    float dout = out_grad[idx];

    // Accumulate dbetta
    partial_betta += dout;

    // Compute xhat and accumulate dgamma
    if (means != nullptr && vars != nullptr) {
      // Use input and per-row mean/var
      float inp_val = inp[idx];
      float mean = means[row];
      float var = vars[row];
      float rstd = rsqrtf(var);
      float xhat = (inp_val - mean) * rstd;
      partial_gamma += xhat * dout;
    } else {
      // Use inp with gamma/betta per-column
      float out_val = inp[idx]; // here inp is actually output because means is nullptr
      float temp_g = gamma[col];
      float temp_b = betta[col];
      // Avoid division by zero if g==0 (unlikely in valid networks)
      float xhat = (out_val - temp_b) / temp_g;
      partial_gamma += xhat * dout;
    }
  }

  // Step 2 Store the partial gradients in the shared memory arrays.
  // Each thread writes its scalar partials into the shared tile at [tx][ty].
  betta_buffer[threadIdx.x][threadIdx.y] = partial_betta;
  gamma_buffer[threadIdx.x][threadIdx.y] = partial_gamma;

  // Ensure all writes to shared memory are visible before reduction reads
  __syncthreads();

  // Step 2 Store the partial gradients in the shared memory arrays.
  // Shared Memory Storage: The computed partial gradient values are 
  // stored in shared memory arrays betta_buffer and gamma_buffer in a tiled manner.
  
  // Step 3 Compute the reduce sum of the shared memory arrays with g.shfl_down.
  float betta_acc = betta_buffer[threadIdx.x][threadIdx.y];
  float gamma_acc = gamma_buffer[threadIdx.x][threadIdx.y];
  for (int i = g.size() / 2; i > 0; i /= 2) {
    betta_acc += g.shfl_down(betta_acc, i);
    gamma_acc += g.shfl_down(gamma_acc, i);
  }

  // Step 4 Assign the final result to the correct position in the global output.
  if (g.thread_rank() == 0) {
    betta_grad[blockIdx.x] = betta_acc;
    gamma_grad[blockIdx.x] = gamma_acc;
  }


  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  // Step 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup

  // Initialization:   // Hint: Each thread is responsible for a specfic element in the inp_grad array
  float4 *dy_f4 = reinterpret_cast<float4 *>(const_cast<T *>(out_grad)) + blockIdx.x * hidden_dim;
  float4 *inp_f4 = reinterpret_cast<float4 *>(const_cast<T *>(inp)) + blockIdx.x * hidden_dim;

  float4 *gamma_f4 = reinterpret_cast<float4 *>(const_cast<T *>(gamma));
  float4 *betta_f4 = reinterpret_cast<float4 *>(const_cast<T *>(betta));

  float4 dxhat;
  
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 dy = dy_f4[idx];
    float4 g = gamma_f4[idx];
    dxhat.x = dy.x * g.x;
    dxhat.y = dy.y * g.y;
    dxhat.z = dy.z * g.z;
    dxhat.w = dy.w * g.w;
  }

  // Step 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  float4 *inp_f4 = reinterpret_cast<float4 *>(const_cast<T *>(inp)) + blockIdx.x * hidden_dim;
  float4 xhat;
  
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    if (means != nullptr && vars != nullptr) {
      // Use input and per-row mean/var
      float mean = means[blockIdx.x];
      float var = vars[blockIdx.x];
      float rstd = rsqrtf(var);

      float4 inp_val = inp_f4[idx];
      xhat.x = (inp_val.x - mean) * rstd;
      xhat.y = (inp_val.y - mean) * rstd;
      xhat.z = (inp_val.z - mean) * rstd;
      xhat.w = (inp_val.w - mean) * rstd;
    } else {
      // Use inp with gamma/betta per-column
      float4 out_val = inp_f4[idx]; // here inp is actually output because means is nullptr
      float4 temp_g = gamma_f4[idx];
      float4 temp_b = betta_f4[idx];
      // Avoid division by zero if g==0 (unlikely in valid networks)
      xhat.x = (out_val.x - temp_b.x) / temp_g.x;
      xhat.y = (out_val.y - temp_b.y) / temp_g.y;
      xhat.z = (out_val.z - temp_b.z) / temp_g.z;
      xhat.w = (out_val.w - temp_b.w) / temp_g.w;
    }
  }

  // Step 3 Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  blockReduce<ReduceType::kSum, 4>(reinterpret_cast<float*>(dxhat));
  blockReduce<ReduceType::kSum, 4>(reinterpret_cast<float*>(xhat));
  __shared__ float s_dxhat;
  __shared__ float s_dxhat_xhat;
  if (threadIdx.x == 0) {
    s_dxhat = reinterpret_cast<float*>(dxhat)[0];
    s_dxhat_xhat = reinterpret_cast<float*>(dxhat_xhat)[0] * reinterpret_cast<float*>(xhat)[0];
  }

  __syncthreads();

  // Step 4 Compute final gradient.
  // dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / (hidden_dim*4)) * rsqrt(var)

  float rstd = rsqrtf(vars[blockIdx.x]);
  int total = hidden_dim * 4; // total number of floats in the row
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    float dy = ((float*)out_grad)[blockIdx.x * total + i];
    float g_val = ((float*)gamma)[i];
    float dxhat = dy * g_val;
    float xhat;
    if (means != nullptr && vars != nullptr) {
        float row_mean = means[blockIdx.x];
        float var = vars[blockIdx.x];
        float rstd_local = rsqrtf(var);
        float inp_val = ((float*)inp)[blockIdx.x * total + i];
        xhat = (inp_val - row_mean) * rstd_local;
    } else {
        float out_val = ((float*)inp)[blockIdx.x * total + i];
        float b = ((float*)betta)[i];
        xhat = (out_val - b) / g_val;
    }
    float result = (dxhat - (s_dxhat + xhat * s_xhat_dxhat) / ((float)total)) * rstd;
    ((float*)inp_grad)[blockIdx.x * total + i] = result;
  }

  
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM); // So we are tiling along the hidden dim
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
