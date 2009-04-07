
#include "buffer.h"

#include <scm/gl/graphics_device/opengl3/device.h>

namespace scm {
namespace gl {

opengl_buffer::opengl_buffer(opengl_device& dev)
  : buffer(dev)
{
}

opengl_buffer::~opengl_buffer()
{
}

} // namespace gl
} // namespace scm
