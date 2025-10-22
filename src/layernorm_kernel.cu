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

/*
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
  /// REFACTORED (To use both Shared Memory and shfl_down)
  // This version attempts to satisfy the contradictory requirements of
  // using both shared memory buffers AND g.shfl_down() for reduction.
  // 1. We use the g.shfl_down() friendly layout (ty -> col, tx -> row_subset).
  // 2. We compute partials in registers.
  // 3. We store those partials in shared memory (as instructed).
  // 4. We read the partials back from shared memory.
  // 5. We reduce the values read from shared memory using g.shfl_down().
  //
  // NOTE: Steps 3 and 4 are redundant as the data is already in registers
  // but are included to follow the assignment instructions literally.

  // Shared memory arrays are declared, as per instructions.
  // The layout is [col_in_block][row_in_block] or [ty][tx]
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  // `g` is a ROW-tile (warp) of TILE_DIM threads.
  // `g.thread_rank()` corresponds to `threadIdx.x`.
  // All threads in `g` have the same `threadIdx.y`.
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1: Compute the partial gradients by looping across inp rows.
  float partial_betta = 0.0f;
  float partial_gamma = 0.0f;

  // Compute column index. All threads in this tile `g` have the same `threadIdx.y`.
  int col = blockIdx.x * TILE_DIM + threadIdx.y;

  // Loop over rows. g.thread_rank() is equivalent to threadIdx.x
  for (int row = g.thread_rank(); row < rows; row += TILE_DIM) {
    if (col >= width) break; // This tile is outside the actual width

    int idx = row * width + col;
    float dout = out_grad[idx];
    partial_betta += dout;

    if (means != nullptr && vars != nullptr) {
      float inp_val = inp[idx];
      float mean = means[row];
      float var = vars[row];
      float rstd = rsqrtf(var);
      float xhat = (inp_val - mean) * rstd;
      partial_gamma += xhat * dout;
    } else {
      float out_val = inp[idx]; // here inp is output
      float temp_g = gamma[col];
      float temp_b = betta[col];
      float xhat = (out_val - temp_b) / temp_g;
      partial_gamma += xhat * dout;
    }
  }

  // Step 2: Store the partial gradients in the shared memory arrays.
  // We store in [ty][tx] layout.
  betta_buffer[threadIdx.y][threadIdx.x] = partial_betta;
  gamma_buffer[threadIdx.y][threadIdx.x] = partial_gamma;

  // All threads must finish writing before *any* thread can read.
  __syncthreads();

  // Step 3: Compute the reduce sum of the shared memory arrays with g.shfl_down.
  // First, read the values back from shared memory.
  // (This is the redundant part, but follows instructions).
  // float betta_acc = betta_buffer[threadIdx.y][threadIdx.x];
  // float gamma_acc = gamma_buffer[threadIdx.y][threadIdx.x];

  // Perform the reduction over the tile 'g' (across threadIdx.x)
  for (int offset = g.size() / 2; offset > 0; offset /= 2) {
    betta_buffer[threadIdx.y][threadIdx.x] += g.shfl_down(betta_buffer[threadIdx.y][threadIdx.x], offset);
    gamma_buffer[threadIdx.y][threadIdx.x] += g.shfl_down(gamma_buffer[threadIdx.y][threadIdx.x], offset);
  }

  // After the loop, thread with rank 0 (tx == 0) in each tile (ty)
  // has the total sum for its column.

  __syncthreads();

  // Step 4: Assign the final result to the correct position
  // Only the root thread of the reduction (rank 0) writes the final result.
  if (g.thread_rank() == 0) {
    // col is (blockIdx.x * TILE_DIM + threadIdx.y)
    if (col < width) {
      betta_grad[col] = betta_buffer[threadIdx.y][threadIdx.x];
      gamma_grad[col] = gamma_buffer[threadIdx.y][threadIdx.x];
    }
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


  // Initialization:   
  float4 *dy_f4 = reinterpret_cast<float4 *>(const_cast<T *>(out_grad)) + blockIdx.x * hidden_dim;
  float4 *inp_f4 = reinterpret_cast<float4 *>(const_cast<T *>(inp)) + blockIdx.x * hidden_dim;
  float4 *out_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;

  float4 *gamma_f4 = reinterpret_cast<float4 *>(const_cast<T *>(gamma));
  float4 *betta_f4 = reinterpret_cast<float4 *>(const_cast<T *>(betta));

  float4 dxhat = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 xhat = {0.0f, 0.0f, 0.0f, 0.0f};

  int idx = threadIdx.x;  // Local index within this row

  // Only process if this thread has valid data
  if (idx < hidden_dim) {
    float4 dy = dy_f4[idx];
    float4 g = gamma_f4[idx];
    dxhat.x = dy.x * g.x;
    dxhat.y = dy.y * g.y;
    dxhat.z = dy.z * g.z;
    dxhat.w = dy.w * g.w;

    // Step 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
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
  // Create local arrays for reduction - compute dxhat * xhat product per thread
  float l_dxhat[4] = {dxhat.x, dxhat.y, dxhat.z, dxhat.w};
  float l_dxhat_xhat[4] = {dxhat.x * xhat.x, dxhat.y * xhat.y,
                            dxhat.z * xhat.z, dxhat.w * xhat.w};

  // Reduce both arrays across all threads in the block
  blockReduce<ReduceType::kSum, 4>(l_dxhat);
  blockReduce<ReduceType::kSum, 4>(l_dxhat_xhat);

  // Store reduced values in shared memory (only thread 0 has the final result)
  // After blockReduce, each of the 4 elements contains the sum across all threads for that component
  __shared__ float sum_dxhat;
  __shared__ float sum_dxhat_xhat;
  
  if (threadIdx.x == 0) {
    // Sum all 4 reduced components to get the total sum across the entire row
    sum_dxhat = l_dxhat[0] + l_dxhat[1] + l_dxhat[2] + l_dxhat[3];
    sum_dxhat_xhat = l_dxhat_xhat[0] + l_dxhat_xhat[1] + l_dxhat_xhat[2] + l_dxhat_xhat[3];
  }

  __syncthreads();

  // Step 4 Compute final gradient.
  // dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim) * rsqrt(var)
  // Note: hidden_dim here is already divided by 4, so total elements = hidden_dim * 4

  float rstd = rsqrtf(vars[blockIdx.x]);
  float total = (float)(hidden_dim * 4); // total number of floats in the row

  if (idx < hidden_dim) {
    out_f4[idx].x = (dxhat.x - (sum_dxhat + xhat.x * sum_dxhat_xhat) / total) * rstd;
    out_f4[idx].y = (dxhat.y - (sum_dxhat + xhat.y * sum_dxhat_xhat) / total) * rstd;
    out_f4[idx].z = (dxhat.z - (sum_dxhat + xhat.z * sum_dxhat_xhat) / total) * rstd;
    out_f4[idx].w = (dxhat.w - (sum_dxhat + xhat.w * sum_dxhat_xhat) / total) * rstd;
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
  // Calculate the number of blocks needed to cover hidden_dim with TILE_DIM threads per block
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM); // Number of blocks along hidden dimension
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
