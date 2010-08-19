
#include "scmgl.h"

#include <scm/core/platform/platform.h>

#include <scm/gl/graphics_device/device.h>
#include <scm/gl/graphics_device/opengl3/device.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#include <scm/gl/graphics_device/opengl3/device_win32.h>
#include <scm/gl/graphics_device/direct3d11/device.h>
#else
//#error "currently unsupported platform"
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS


namespace scm {
namespace gl {

device_initializer::device_initializer()
  : _type(SCMGL_DEVICE_NULL)
{
}

device_context_config::device_context_config()
  : _output_dimensions(800u, 600u),
    _color_buffer_format(FORMAT_RGBA8),
    _depth_stencil_buffer_format(FORMAT_D24_S8),
    _color_buffer_count(2),
    _sample_count(1),
    _context_type(CONTEXT_WINDOWED),
    _refresh_rate(60)
{
}

device_ptr
create_device(const device_initializer& init,
              const device_context_config& cfg)
{
    switch (init._type) {
#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
        case SCMGL_DEVICE_OPENGL:   return device_ptr(new opengl_device_win32(init, cfg));break;
        case SCMGL_DEVICE_DIREC3D:  return device_ptr(new direct3d_device(init, cfg));break;
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
        case SCMGL_DEVICE_NULL:     break;
    }

    return (device_ptr());
}


} // namespace gl
} // namespace scm
