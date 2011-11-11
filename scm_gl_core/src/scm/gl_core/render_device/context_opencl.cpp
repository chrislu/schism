
#include "context.h"

#include <CL/cl.hpp>

#include <scm/gl_core/config.h>
#include <scm/gl_core/config_cl.h>
#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/opencl_interop.h>

namespace scm {
namespace gl {
    
const cl::command_queue_ptr&
render_context::cl_command_queue() const
{
    return _cl_command_queue;
}

bool
render_context::init_opencl(render_device& in_device)
{
#if SCM_GL_CORE_OPENCL_ENABLE_INTEROP
    using scm::cl::util::cl_error_string;

    cl_int          cl_error = CL_SUCCESS;

    { // create command queue
        cl_command_queue_properties cmd_prop = 0;
#if SCM_GL_CORE_OPENCL_ENABLE_PROFILING
        cmd_prop = CL_QUEUE_PROFILING_ENABLE;
#endif // SCM_GL_CORE_OPENCL_ENABLE_PROFILING
        _cl_command_queue.reset(new cl::CommandQueue(*in_device.cl_context(), *in_device.cl_device(), cmd_prop, &cl_error));
        if (CL_SUCCESS != cl_error) {
            err() << log::error
                  << "render_context::init_opencl(): "
                  << "unable to create OpenCL command queue"
                  << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
    }
#endif // SCM_GL_CORE_OPENCL_ENABLE_INTEROP

    return true;
}

} // namespace gl
} // namespace scm
