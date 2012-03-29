
#ifndef SCM_LARGE_DATA_OPENCL_VOLUME_RENDERER_H_INCLUDED
#define SCM_LARGE_DATA_OPENCL_VOLUME_RENDERER_H_INCLUDED

#include <vector>

#include <scm/core/math.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/cl_core/opencl/opencl_fwd.h>
#include <scm/cl_core/opencl/accum_timer.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

#include <renderer/renderer_fwd.h>

namespace scm {
namespace data {

class opencl_volume_renderer
{
public:
    typedef time::accum_timer<time::high_res_timer> cpu_timer_type;

public:
    opencl_volume_renderer(const gl::render_device_ptr& device,
                           const cl::opencl_device_ptr& cl_device,
                           const math::vec2ui&          viewport_size);
    virtual ~opencl_volume_renderer();

    void                            draw(const gl::render_context_ptr& context,
                                         const cl::command_queue_ptr&  cl_queue,
                                         const opencl_volume_data_ptr& vdata);
    void                            present(const gl::render_context_ptr& context) const;

    bool                            reload_kernels(const gl::render_device_ptr& device,
                                                   const cl::opencl_device_ptr& cl_device);

protected:
    void                            cleanup();

protected:
    math::vec2ui                    _viewport_size;

    gl::texture_2d_ptr              _output_texture;
    cl::image_2d_gl_ptr             _output_cl_image;

    cl::kernel_ptr                  _ray_cast_kernel;
    math::vec2ui                    _ray_cast_kernel_wg_size;
    std::vector<cl::Memory>         _shared_gl_objects;

    gl::texture_output_ptr          _texture_presenter;

    cpu_timer_type                  _acquire_timer;
    cpu_timer_type                  _release_timer;
    cpu_timer_type                  _kernel_timer;

    cl::util::accum_timer           _cl_acquire_timer;
    cl::util::accum_timer           _cl_release_timer;
    cl::util::accum_timer           _cl_kernel_timer;

}; // class opencl_volume_renderer

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_OPENCL_VOLUME_RENDERER_H_INCLUDED
