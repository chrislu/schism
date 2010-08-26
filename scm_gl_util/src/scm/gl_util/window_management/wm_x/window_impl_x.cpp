
#include "window_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

namespace scm {
namespace gl {
namespace wm {

window::window_impl::window_impl(const display&           in_display,
                                 const std::string&       in_title,
                                 const math::vec2ui&      in_size,
                                 const pixel_format_desc& in_pf)
{
}

window::window_impl::~window_impl()
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
