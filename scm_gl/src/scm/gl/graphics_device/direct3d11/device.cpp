
#include "device.h"

namespace scm {
namespace gl {

direct3d_device::direct3d_device(const device_initializer& init,
                                 const device_context_config& cfg)
{
}

direct3d_device::~direct3d_device()
{
}

bool
direct3d_device::setup_render_context(const device_context_config& cfg,
                                      unsigned                    feature_level)
{
    return (false);
}

} // namespace gl
} // namespace scm
