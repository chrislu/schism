
#include "device_resource.h"

#include <scm/gl/graphics_device/device.h>

namespace scm {
namespace gl {

device_resource::device_resource(device& owning_device)
  : _owning_device(owning_device)
{
}

device_resource::~device_resource()
{
}

const device&
device_resource::owning_device() const
{
    return (_owning_device);
}

} // namespace gl
} // namespace scm
