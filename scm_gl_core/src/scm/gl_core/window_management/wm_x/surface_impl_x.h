
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_X_SURFACE_IMPL_X_H_INCLUDED
#define SCM_GL_CORE_WM_X_SURFACE_IMPL_X_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>
#include <GL/glx.h>

#include <scm/gl_core/window_management/surface.h>

namespace scm {
namespace gl {
namespace wm {

struct surface::surface_impl
{
    surface_impl() : _fb_config(0), _drawable(0) {}

    GLXFBConfig     _fb_config;
    GLXDrawable     _drawable;
}; // class surface_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_CORE_WM_X_SURFACE_IMPL_X_H_INCLUDED
