
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LARGE_DATA_CUDA_VOLUME_RAY_CAST_KERNEL_H_INCLUDED
#define SCM_LARGE_DATA_CUDA_VOLUME_RAY_CAST_KERNEL_H_INCLUDED

#include <renderer/volume_uniform_data.h>

namespace scm {
namespace cuda {

void
startup_ray_cast_kernel(unsigned out_image_w, unsigned out_image_h,
                        cudaGraphicsResource_t                   output_image_res,
                        cudaGraphicsResource_t                   volume_image_res,
                        cudaGraphicsResource_t                   cmap_image_res,
                        cudaStream_t                             cuda_stream);

bool
upload_uniform_data(const volume_uniform_data& vud,
                    cudaStream_t               cuda_stream);

} // namespace cuda
} // namespace scm

#endif // SCM_LARGE_DATA_CUDA_VOLUME_RAY_CAST_KERNEL_H_INCLUDED
