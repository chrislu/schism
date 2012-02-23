
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "display_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>

#include <scm/gl_core/window_management/wm_x/util/glx_extensions.h>

namespace scm {
namespace gl {
namespace wm {

display::display_impl::display_impl(const std::string& name)
  : _display(0)
{
    try {
        _display = ::XOpenDisplay(name.c_str());
        if (0 == _display) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <xlib>: "
              << "unable to open display (" << name << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        _default_screen = ::XDefaultScreen(_display);

        std::stringstream glx_err;
        _glx_extensions.reset(new util::glx_extensions());
        if (!_glx_extensions->initialize(_display, glx_err)) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <xlib>: "
              << "unable to initialize GLX ARB extensions, GLX init failed: "
              << glx_err.str();
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }
    }
    catch(...) {
        cleanup();
        throw;
    }
}

display::display_impl::~display_impl()
{
    cleanup();
}

void
display::display_impl::cleanup()
{
    if (_display) {
        XCloseDisplay(_display);
    }
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
