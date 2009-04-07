
#include "device.h"

namespace scm {
namespace gl {

device::device()
  : _type(SCMGL_DEVICE_NULL),
    _feature_level(0)
{
}

device::~device()
{
}

unsigned
device::feature_level() const
{
    return (_feature_level);
}

device_type
device::type() const
{
    return (_type);
}

} // namespace gl
} // namespace scm
