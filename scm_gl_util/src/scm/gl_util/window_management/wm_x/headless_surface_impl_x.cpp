
#include "headless_surface_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/window.h>

namespace scm {
namespace gl {
namespace wm {

namespace detail {

const int fixed_pbuffer_width   = 1;
const int fixed_pbuffer_height  = 1;

} // namespace detail

headless_surface::headless_surface_impl::headless_surface_impl(const window_ptr& in_parent_wnd)
  : surface::surface_impl()
{
    try {
    }
    catch(...) {
        cleanup();
        throw;
    }
}

headless_surface::headless_surface_impl::~headless_surface_impl()
{
    cleanup();
}

void
headless_surface::headless_surface_impl::cleanup()
{
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
