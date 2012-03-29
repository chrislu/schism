
#include "opencl_volume_data.h"

#include <exception>
#include <sstream>
#include <stdexcept>

#include <CL/cl.hpp>

#include <scm/log.h>

#include <scm/core/log/logger_state.h>

#include <scm/cl_core/opencl.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/viewer/camera.h>

#include <renderer/volume_data.h>

namespace scm {
namespace data {

opencl_volume_data::opencl_volume_data(const gl::render_device_ptr& device,
                                       const cl::opencl_device_ptr& cl_device,
                                       const volume_data_ptr&       voldata)
  : _data(voldata)
{
    cl_int cl_error     = CL_SUCCESS;
    
    log::logger_format_saver out_save(out().associated_logger());
    out() << "opencl_volume_data::opencl_volume_data(): creating OpenCL volume data resources..." << log::end;
    out() << log::indent;
    {
        std::stringstream   os;
        device->dump_memory_info(os);
        out() << log::info
              << "before creating volume gl image:" << log::nline
              << os.str() << log::end;
    }
    _volume_image.reset(new cl::Image3DGL(*cl_device->cl_context(), CL_MEM_READ_ONLY,
                                            voldata->volume_raw()->object_target(), 0,
                                            voldata->volume_raw()->object_id(), &cl_error));
    if (CL_SUCCESS != cl_error) {
        _volume_image.reset();
        std::stringstream os;
        os << "opencl_volume_data::opencl_volume_data(): "
            << "error creating cl image from volume_raw texture (" << cl::util::cl_error_string(cl_error) << ")." << std::endl;
        throw std::runtime_error(os.str());
    }
    {
        std::stringstream   os;
        device->dump_memory_info(os);
        out() << log::info
              << "after creating volume gl image:" << log::nline
              << os.str() << log::end;
    }

    _color_alpha_image.reset(new cl::Image2DGL(*cl_device->cl_context(), CL_MEM_READ_ONLY,
                                               voldata->color_alpha_map()->object_target(), 0,
                                               voldata->color_alpha_map()->object_id(), &cl_error));
    if (CL_SUCCESS != cl_error) {
        _color_alpha_image.reset();
        std::stringstream os;
        os << "opencl_volume_data::opencl_volume_data(): "
           << "error creating cl image from color_alpha_map texture (" << cl::util::cl_error_string(cl_error) << ")." << std::endl;
        throw std::runtime_error(os.str());
    }

    //_volume_uniform_buffer.reset(new cl::Buffer(*device->cl_context(),
    //                                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //                                            sizeof(volume_uniform_data),
    //                                            voldata->volume_block().get_block(), &cl_error));
    _volume_uniform_buffer.reset(new cl::Buffer(*cl_device->cl_context(),
                                                CL_MEM_READ_ONLY,
                                                sizeof(volume_uniform_data),
                                                0, &cl_error));
    if (CL_SUCCESS != cl_error) {
        _volume_uniform_buffer.reset();
        std::stringstream os;
        os << "opencl_volume_data::opencl_volume_data(): "
           << "unable to create volume_uniform_buffer buffer"
           << "(" << cl::util::cl_error_string(cl_error) << ")." << std::endl;
        throw std::runtime_error(os.str());
    }
}

opencl_volume_data::~opencl_volume_data()
{
    _data.reset();
    _volume_image.reset();
    _color_alpha_image.reset();
    _volume_uniform_buffer.reset();
}

void
opencl_volume_data::update(const gl::render_context_ptr& context,
                           const cl::command_queue_ptr& cl_queue)
{
    cl_int cl_error = CL_SUCCESS;
    //scm::size_t s = sizeof(volume_uniform_data);

    volume_uniform_data d;
    memcpy(&d, data()->volume_block().get_block(), sizeof(volume_uniform_data));

    d._m_matrix                    = math::transpose(d._m_matrix                   );
    d._m_matrix_inverse            = math::transpose(d._m_matrix_inverse           );
    d._m_matrix_inverse_transpose  = math::transpose(d._m_matrix_inverse_transpose );
    d._mv_matrix                   = math::transpose(d._mv_matrix                  );
    d._mv_matrix_inverse           = math::transpose(d._mv_matrix_inverse          );
    d._mv_matrix_inverse_transpose = math::transpose(d._mv_matrix_inverse_transpose);
    d._mvp_matrix                  = math::transpose(d._mvp_matrix                 );
    d._mvp_matrix_inverse          = math::transpose(d._mvp_matrix_inverse         );

    cl_error = cl_queue->enqueueWriteBuffer(*_volume_uniform_buffer, false,
                                            0, sizeof(volume_uniform_data),
                                            &d, //data()->volume_block().get_block(),
                                            0, 0);

    //context->cl_command_queue()->finish();
    assert(!scm::cl::util::cl_error_string(cl_error).empty());
}

const volume_data_ptr&
opencl_volume_data::data() const
{
    return _data;
}

const cl::image_3d_gl_ptr&
opencl_volume_data::volume_image() const
{
    return _volume_image;
}

const cl::image_2d_gl_ptr&
opencl_volume_data::color_alpha_image() const
{
    return _color_alpha_image;
}

const cl::buffer_ptr&
opencl_volume_data::volume_uniform_buffer() const
{
    return _volume_uniform_buffer;
}

} // namespace data
} // namespace scm
