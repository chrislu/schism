
#ifndef SCM_LARGE_DATA_CUDA_VOLUME_RENDERER_H_INCLUDED
#define SCM_LARGE_DATA_CUDA_VOLUME_RENDERER_H_INCLUDED

#include <vector>

#include <cuda_runtime.h>

#include <scm/core/math.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

#include <renderer/renderer_fwd.h>

namespace scm {
namespace data {

class cuda_volume_renderer
{
public:
    typedef time::accum_timer<time::high_res_timer> cpu_timer_type;

public:
    cuda_volume_renderer(const gl::render_device_ptr& device,
                         const math::vec2ui&          viewport_size);
    virtual ~cuda_volume_renderer();

    void                            draw(const gl::render_context_ptr& context,
                                         const cuda_volume_data_ptr& vdata);
    void                            present(const gl::render_context_ptr& context) const;

protected:
    void                            cleanup();

protected:
    math::vec2ui                    _viewport_size;

    gl::texture_2d_ptr              _output_texture;

    gl::texture_output_ptr          _texture_presenter;

    cpu_timer_type                  _acquire_timer;
    cpu_timer_type                  _release_timer;
    cpu_timer_type                  _kernel_timer;

    shared_ptr<cudaGraphicsResource>    _output_image;

}; // class cuda_volume_renderer

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_CUDA_VOLUME_RENDERER_H_INCLUDED
