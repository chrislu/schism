
#include "device_context.h"

#include <scm/gl/graphics_device/opengl3/device.h>

namespace scm {
namespace gl {

opengl_device_context::opengl_device_context(opengl_device& dev)
  : device_context(dev)
{
}

opengl_device_context::~opengl_device_context()
{
}

const handle
opengl_device_context::context_handle() const
{
    return (_context_handle);
}


} // namespace gl
} // namespace scm
