
#include "cuda/gaussian_blur.h"

#include <scm/ogl/gl.h>
#include <cuda_gl_interop.h>

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgb_to_int(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

// get pixel from 2D image, with clamping to border
__device__ int get_pixel(int *data, int x, int y, int width, int height)
{
    x = clamp(x, 0, width-1);
    y = clamp(y, 0, height-1);
    return data[y*width+x];
}

static const unsigned   kernel_radius = 3u;                     // 7x7 gauss
static const unsigned   kernel_width  = 2 * kernel_radius + 1u; // 7x7 gauss

__device__ __constant__ float gauss_kernel_constant[kernel_width * kernel_width];

__global__ void cuda_gaussian_blur_7x7(int* in_data, int* out_data, unsigned width, unsigned height)
{
    extern __shared__ int shared_memory[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

#if 1
    int2 shmem_tile_size = make_int2(ceilf(float(blockDim.x + 2 * kernel_radius) / float(blockDim.x)),
                                     ceilf(float(blockDim.y + 2 * kernel_radius) / float(blockDim.y)));

    int2 shmem_size      = make_int2(blockDim.x + 2 * kernel_radius,
                                     blockDim.y + 2 * kernel_radius);


    for (int sy = threadIdx.y * shmem_tile_size.y;
             sy < clamp((threadIdx.y + 1) * shmem_tile_size.y, 0, shmem_size.y);
           ++sy) {
        for (int sx = threadIdx.x * shmem_tile_size.x;
                 sx < clamp((threadIdx.x + 1) * shmem_tile_size.x, 0, shmem_size.x);
               ++sx) {
            shared_memory[sy * shmem_size.y + sx] = get_pixel(in_data,
                                                              sx + blockIdx.x * blockDim.x - kernel_radius,
                                                              sy + blockIdx.y * blockDim.y - kernel_radius,
                                                              width,
                                                              height);
        }
    }

    // wait for threads to complete loading of shared memory
    __syncthreads();

    // use shared memory for access to pixel fetches
    float3 out_color = make_float3(0.0f, 0.0f, 0.0f);

    for(int dy = 0; dy < kernel_width; ++dy) {
        for (int dx = 0; dx < kernel_width; ++dx) {
            
            int     pixel   = shared_memory[(dy +  threadIdx.y) * shmem_size.y + dx + threadIdx.x];

            float   weight  = gauss_kernel_constant[dy*kernel_width + dx];

            float3 in_color = make_float3(float(pixel&0xff),
                                          float((pixel>>8)&0xff),
                                          float((pixel>>16)&0xff));

            out_color.x += in_color.x * weight;
            out_color.y += in_color.y * weight;
            out_color.z += in_color.z * weight;
        }
    }

    out_data[y * width + x] = rgb_to_int(out_color.x,
                                         out_color.y,
                                         out_color.z);
#endif

#if 0
    // use global memory accesses only for pixel fetches
    float3 out_color = make_float3(0.0f, 0.0f, 0.0f);

    for(int dy = 0; dy < kernel_width; ++dy) {
        for (int dx = 0; dx < kernel_width; ++dx) {
            int     pixel   = get_pixel(in_data,
                                        x + dx - kernel_radius,
                                        y + dy - kernel_radius,
                                        width,
                                        height);
            float   weight  = gauss_kernel_constant[dy*kernel_width + dx];

            float3 in_color = make_float3(float(pixel&0xff),
                                          float((pixel>>8)&0xff),
                                          float((pixel>>16)&0xff));

            out_color.x += in_color.x * weight;
            out_color.y += in_color.y * weight;
            out_color.z += in_color.z * weight;
        }
    }

    out_data[y * width + x] = rgb_to_int(out_color.x,
                                         out_color.y,
                                         out_color.z);
#endif

#if 0
    // simple color inversion for testing purposes
    int    pixel     = get_pixel(in_data, x, y, width, height);
    float3 out_color = make_float3(float(pixel&0xff),
                                   float((pixel>>8)&0xff),
                                   float((pixel>>16)&0xff));
    out_data[y * width + x] = rgb_to_int(255.0f - out_color.x,
                                         255.0f - out_color.y,
                                         255.0f - out_color.z);
#endif
}

void gaussian_blur_7x7(int* in_data, int* out_data, unsigned width, unsigned height)
{
    if (!cudaMemcpyToSymbol(gauss_kernel_constant, gauss_kernel_7x7, kernel_width * kernel_width * sizeof(float)) != cudaSuccess) {
        //
    }

    dim3 block_size(16, 16, 1);
    dim3 grid_size(width / block_size.x, height / block_size.y, 1);
    int  shared_mem_size =   (block_size.x + 2 * kernel_radius)
                           * (block_size.y + 2 * kernel_radius)
                           * sizeof(int);

    cuda_gaussian_blur_7x7<<<grid_size, block_size, shared_mem_size>>>(in_data, out_data, width, height);
}
