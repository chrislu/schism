
#ifndef SCM_LARGE_DATA_CUDA_VOLUME_RAY_CAST_KERNEL_H_INCLUDED
#define SCM_LARGE_DATA_CUDA_VOLUME_RAY_CAST_KERNEL_H_INCLUDED

#include <thrust/device_ptr.h>
#include <renderer/volume_uniform_data.h>

extern "C"
void
startup_ray_cast_kernel(unsigned out_image_w, unsigned out_image_h,
                        cudaGraphicsResource_t                   output_image_res,
                        cudaGraphicsResource_t                   volume_image_res,
                        cudaGraphicsResource_t                   cmap_image_res,
                        const thrust::device_ptr<volume_uniform_data>& uniform_data,
                        cudaStream_t                             cuda_stream);

#endif // SCM_LARGE_DATA_CUDA_VOLUME_RAY_CAST_KERNEL_H_INCLUDED
