
#include "render_target.h"

#include <scm/gl_core/render_device.h>

namespace scm {
namespace gl {

render_target::render_target(render_device& in_device)
  : render_device_resource(in_device),
    _gl_object_id(0),
    _gl_object_target(0)
{
}

render_target::~render_target()
{
}

unsigned
render_target::object_id() const
{
    return (_gl_object_id);
}

unsigned
render_target::object_target() const
{
    return (_gl_object_target);
}


} // namespace gl
} // namespace scm
