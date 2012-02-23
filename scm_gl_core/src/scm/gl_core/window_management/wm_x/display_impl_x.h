
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_X_DISPLAY_IMPL_H_INCLUDED
#define SCM_GL_CORE_WM_X_DISPLAY_IMPL_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/display.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

class glx_extensions;

} // namespace util

struct display::display_impl
{
    display_impl(const std::string& name);
    virtual ~display_impl();

    void                        cleanup();

    ::Display*      _display;
    int             _default_screen;

    shared_ptr<util::glx_extensions>  _glx_extensions;

}; // class display_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_CORE_WM_X_DISPLAY_IMPL_H_INCLUDED
