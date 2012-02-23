
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_X_WINDOW_IMPL_H_INCLUDED
#define SCM_GL_CORE_WM_X_WINDOW_IMPL_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>
#include <GL/glx.h>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/window.h>
#include <scm/gl_core/window_management/wm_x/surface_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

class glx_extensions;

} // namespace util

struct window::window_impl : public surface::surface_impl
{
    window_impl(const display_cptr&      in_display,
                const ::Window           in_parent,
                const std::string&       in_title,
                const math::vec2i&       in_position,
                const math::vec2ui&      in_size,
                const format_desc&       in_sf);
    virtual ~window_impl();

    void            cleanup();

    void            swap_buffers(int interval) const;

    void            show();
    void            hide();

    ::Window        _window_handle;
    ::GLXWindow     _glx_window_handle;

    ::Display*const _display;

    shared_ptr<util::glx_extensions>  _glx_extensions;

}; // class window_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_CORE_WM_X_WINDOW_IMPL_H_INCLUDED
