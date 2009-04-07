
#include "device_context.h"

#include <scm/gl/graphics_device/direct3d11/device.h>

namespace scm {
namespace gl {

direct3d_device_context::direct3d_device_context(direct3d_device& dev)
  : device_context(dev)
{
}

direct3d_device_context::~direct3d_device_context()
{
}

} // namespace gl
} // namespace scm
