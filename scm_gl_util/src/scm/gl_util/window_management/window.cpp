
#include "window.h"

#include <exception>
#include <stdexcept>

#include <scm/log.h>

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/pixel_format.h>
#include <scm/gl_util/window_management/wm_win32/window_impl_win32.h>
#include <scm/gl_util/window_management/wm_x/window_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

window::window(const display&           in_display,
               const std::string&       in_title,
               const math::vec2i&       in_position,
               const math::vec2ui&      in_size,
               const pixel_format_desc& in_pf)
  : surface(in_display, in_pf)
{
    try {
        _impl.reset(new window_impl(in_display, in_title, in_position, in_size, in_pf));
    }
    catch(const std::exception& e) {
        err() << log::error
              << "window::window(): "
              << log::indent << e.what() << log::outdent << log::end;
        throw (e);
    }
}

window::~window()
{
    _impl.reset();
}

const window::wnd_handle
window::window_handle() const
{
    return (_impl->_window_handle);
}

void
window::show()
{
    _impl->show();
}

void
window::hide()
{
    _impl->hide();
}

} // namespace wm
} // namepspace gl
} // namepspace scm
