
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "window.h"

#include <exception>
#include <stdexcept>

#include <scm/log.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/context.h>
#include <scm/gl_core/window_management/wm_win32/window_impl_win32.h>
#include <scm/gl_core/window_management/wm_x/window_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

window::window(const display_cptr&      in_display,
               const std::string&       in_title,
               const math::vec2i&       in_position,
               const math::vec2ui&      in_size,
               const format_desc&       in_sf)
  : surface(in_display, in_sf)
{
    try {
        _impl.reset(new window_impl(in_display, 0, in_title, in_position, in_size, in_sf));
    }
    catch(const std::exception& e) {
        err() << log::error
              << "window::window(): "
              << log::indent << e.what() << log::outdent << log::end;
        throw e;
    }
}

window::window(const display_cptr&      in_display,
               const handle             in_parent,
               const std::string&       in_title,
               const math::vec2i&       in_position,
               const math::vec2ui&      in_size,
               const format_desc&       in_sf)
  : surface(in_display, in_sf)
{
    try {
        _impl.reset(new window_impl(in_display, in_parent, in_title, in_position, in_size, in_sf));
    }
    catch(const std::exception& e) {
        err() << log::error
              << "window::window(): "
              << log::indent << e.what() << log::outdent << log::end;
        throw e;
    }
}

window::~window()
{
    _impl.reset();
}

void
window::swap_buffers(int interval) const
{
    scm::static_pointer_cast<window_impl>(_impl)->swap_buffers(interval);
}

const window::handle
window::window_handle() const
{
    return scm::static_pointer_cast<window_impl>(_impl)->_window_handle;
}

void
window::show()
{
    scm::static_pointer_cast<window_impl>(_impl)->show();
}

void
window::hide()
{
    scm::static_pointer_cast<window_impl>(_impl)->hide();
}

} // namespace wm
} // namepspace gl
} // namepspace scm
