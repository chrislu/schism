
#include "window_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

namespace scm {
namespace gl {
namespace wm {

window::window_impl::window_impl(const display_ptr&       in_display,
                                 const XID                in_parent,
                                 const std::string&       in_title,
                                 const math::vec2i&       in_position,
                                 const math::vec2ui&      in_size,
                                 const format_desc&       in_sf)
  : surface::surface_impl()
{
}

window::window_impl::~window_impl()
{
}

void
window::window_impl::swap_buffers(int interval) const
{
}

void
window::window_impl::show()
{
}

void
window::window_impl::hide()
{
}


} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
