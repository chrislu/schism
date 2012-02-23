
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WIN32_HEADLESS_SURFACE_IMPL_WIN32_H_INCLUDED
#define SCM_GL_CORE_WM_WIN32_HEADLESS_SURFACE_IMPL_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>
#include <GL/GL.h>
#include <scm/gl_core/window_management/GL/wglext.h>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/headless_surface.h>
#include <scm/gl_core/window_management/window.h>
#include <scm/gl_core/window_management/wm_win32/surface_impl_win32.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

class wgl_extensions;

} // namespace util

struct headless_surface::headless_surface_impl : public surface::surface_impl
{
    headless_surface_impl(const window_cptr& in_parent_wnd);
    virtual ~headless_surface_impl();

    void            cleanup();

    HPBUFFERARB     _pbuffer_handle;

    shared_ptr<util::wgl_extensions>  _wgl_extensions;

}; // class headless_surface_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CORE_WM_WIN32_HEADLESS_SURFACE_IMPL_WIN32_H_INCLUDED
