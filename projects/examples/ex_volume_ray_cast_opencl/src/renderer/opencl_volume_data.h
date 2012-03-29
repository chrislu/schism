
#ifndef SCM_LARGE_DATA_OPENCL_VOLUME_DATA_H_INCLUDED
#define SCM_LARGE_DATA_OPENCL_VOLUME_DATA_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/cl_core/opencl/opencl_fwd.h>

#include <scm/gl_util/viewer/viewer_fwd.h>

#include <renderer/renderer_fwd.h>
#include <renderer/volume_uniform_data.h>

namespace scm {
namespace data {

class opencl_volume_data
{
public:
    opencl_volume_data(const gl::render_device_ptr& device,
                       const cl::opencl_device_ptr& cl_device,
                       const volume_data_ptr&       voldata);
    virtual ~opencl_volume_data();

    void                            update(const gl::render_context_ptr& context,
                                           const cl::command_queue_ptr& cl_queue);

    const volume_data_ptr&          data() const;

    const cl::image_3d_gl_ptr&      volume_image() const;
    const cl::image_2d_gl_ptr&      color_alpha_image() const;

    const cl::buffer_ptr&           volume_uniform_buffer() const;

protected:
    volume_data_ptr                 _data;

    cl::image_3d_gl_ptr             _volume_image;
    cl::image_2d_gl_ptr             _color_alpha_image;

    cl::buffer_ptr                  _volume_uniform_buffer;

}; // opencl_volume_data

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_OPENCL_VOLUME_DATA_H_INCLUDED
