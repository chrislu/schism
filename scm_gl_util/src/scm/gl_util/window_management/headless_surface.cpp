
#include "headless_surface.h"

#include <exception>
#include <stdexcept>

#include <scm/log.h>

#include <scm/gl_util/window_management/window.h>
#include <scm/gl_util/window_management/pixel_format.h>
#include <scm/gl_util/window_management/wm_win32/headless_surface_impl_win32.h>
#include <scm/gl_util/window_management/wm_win32/window_impl_win32.h>
#include <scm/gl_util/window_management/wm_x/window_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

headless_surface::headless_surface(const window& in_parent_wnd)
  : surface(in_parent_wnd.associated_display(), in_parent_wnd.pixel_format())
{
    try {
        _impl.reset(new headless_surface_impl(in_parent_wnd));
    }
    catch(const std::exception& e) {
        err() << log::error
              << "headless_surface::headless_surface(): "
              << log::indent << e.what() << log::outdent << log::end;
        throw (e);
    }
}

headless_surface::~headless_surface()
{
    _impl.reset();
}

} // namespace wm
} // namepspace gl
} // namepspace scm
