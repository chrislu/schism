
#include "buffer.h"

#include <scm/gl/graphics_device/device.h>

namespace scm {
namespace gl {

buffer::buffer(device& dev)
  : device_resource(dev)
{
}

buffer::~buffer()
{
}

const buffer_descriptor&
buffer::descriptor() const
{
    return (_descriptor);
}

} // namespace gl
} // namespace scm
