
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_X_FRAMEBUFFER_CONFIG_SELECTION_X_H_INCLUDED
#define SCM_GL_CORE_WM_X_FRAMEBUFFER_CONFIG_SELECTION_X_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>
#include <GL/glx.h>

#include <ostream>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/surface.h>

namespace scm {
namespace gl {
namespace wm {

class display;
struct pixel_format_desc;

namespace util {

class glx_extensions;

class framebuffer_config_selector
{
public:
    typedef enum {
        window_surface,
        pbuffer_surface
    } surface_type;

    static GLXFBConfig choose(::Display*const                   in_display,
                              const surface::format_desc&       in_pfd,
                              const surface_type                in_surface_type,
                              const shared_ptr<glx_extensions>& in_glx,
                              std::ostream&                     out_stream);

}; // class framebuffer_config_selector

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_CORE_WM_X_FRAMEBUFFER_CONFIG_SELECTION_X_H_INCLUDED
