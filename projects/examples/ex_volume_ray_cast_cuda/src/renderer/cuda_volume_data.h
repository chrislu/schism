
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LARGE_DATA_CUDA_VOLUME_DATA_H_INCLUDED
#define SCM_LARGE_DATA_CUDA_VOLUME_DATA_H_INCLUDED

#include <cuda_runtime.h>

#include <scm/core/math.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_core/render_device/render_device_fwd.h>

#include <scm/gl_util/viewer/viewer_fwd.h>

#include <renderer/renderer_fwd.h>
#include <renderer/volume_uniform_data.h>

namespace scm {
namespace data {

class cuda_volume_data
{
public:
    cuda_volume_data(const gl::render_device_ptr& device,
                     const volume_data_ptr&       voldata);
    virtual ~cuda_volume_data();

    void                                            update(const gl::render_context_ptr& context);

    const volume_data_ptr&                          data() const;

    const shared_ptr<cudaGraphicsResource>&         volume_image() const;
    const shared_ptr<cudaGraphicsResource>&         color_alpha_image() const;

protected:
    volume_data_ptr                                 _data;

    shared_ptr<cudaGraphicsResource>                _volume_image;
    shared_ptr<cudaGraphicsResource>                _color_alpha_image;

}; // cuda_volume_data

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_CUDA_VOLUME_DATA_H_INCLUDED
