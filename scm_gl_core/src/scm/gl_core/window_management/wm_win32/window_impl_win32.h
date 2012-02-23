
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WIN32_WINDOW_IMPL_WIN32_H_INCLUDED
#define SCM_GL_CORE_WM_WIN32_WINDOW_IMPL_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/window.h>
#include <scm/gl_core/window_management/wm_win32/surface_impl_win32.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

class wgl_extensions;

} // namespace util

struct window::window_impl : public surface::surface_impl
{
    window_impl(const display_cptr&      in_display,
                const HWND               in_parent,
                const std::string&       in_title,
                const math::vec2i&       in_position,
                const math::vec2ui&      in_size,
                const format_desc&       in_sf);
    virtual ~window_impl();

    void            cleanup();

    void            swap_buffers(int interval) const;

    void            show();
    void            hide();

    HWND            _window_handle;

    shared_ptr<util::wgl_extensions>  _wgl_extensions;

}; // class window_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CORE_WM_WIN32_WINDOW_IMPL_WIN32_H_INCLUDED
