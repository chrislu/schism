
#include "device_context.h"

#include <scm/gl/graphics_device/device.h>

namespace scm {
namespace gl {

device_context::device_context(device& dev)
  : device_resource(dev)
{
}

device_context::~device_context()
{
}

} // namespace gl
} // namespace scm
