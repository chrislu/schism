
#include "buffer.h"

#include <scm/gl/graphics_device/direct3d11/device.h>

namespace scm {
namespace gl {

direct3d_buffer::direct3d_buffer(direct3d_device& dev)
  : buffer(dev)
{
}

direct3d_buffer::~direct3d_buffer()
{
}

} // namespace gl
} // namespace scm
