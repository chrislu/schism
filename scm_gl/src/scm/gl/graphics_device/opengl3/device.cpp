
#include "device.h"

#include <exception>
#include <stdexcept>
#include <vector>

#include <scm/log.h>


namespace scm {
namespace gl {

namespace detail {
} // namespace detail

opengl_device::opengl_device(const device_initializer& init,
                             const device_context_config& cfg)
{
    // check if opengl is available

    // check if requested feature level is supported
    // if not throw runtime_error or something
}

opengl_device::~opengl_device()
{
}

//bool
//opengl_device::setup_render_context(const device_output_format& fmt)
//{
//    return (false);
//}

//bool
//opengl_device::setup_render_context(const device_output_format& fmt, unsigned feature_level)
//{
//    return (false);
//}

} // namespace gl
} // namespace scm
