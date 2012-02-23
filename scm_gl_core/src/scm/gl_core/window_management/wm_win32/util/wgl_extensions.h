
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WIN32_WGL_EXTENSIONS_H_INCLUDED
#define SCM_GL_CORE_WM_WIN32_WGL_EXTENSIONS_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <ostream>
#include <set>
#include <string>

#include <scm/core/platform/windows.h>
#include <GL/GL.h>
#include <scm/gl_core/window_management/GL/wglext.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

class wgl_extensions
{
private:
    typedef std::set<std::string> string_set;

public:
    // WGL_ARB_create_context
    PFNWGLCREATECONTEXTATTRIBSARBPROC       wglCreateContextAttribsARB;
    
    // WGL_ARB_pixel_format
    PFNWGLGETPIXELFORMATATTRIBIVARBPROC     wglGetPixelFormatAttribivARB;
    PFNWGLGETPIXELFORMATATTRIBFVARBPROC     wglGetPixelFormatAttribfvARB;
    PFNWGLCHOOSEPIXELFORMATARBPROC          wglChoosePixelFormatARB;

    // WGL_ARB_extensions_string
    PFNWGLGETEXTENSIONSSTRINGARBPROC        wglGetExtensionsStringARB;

    // ARB_framebuffer_sRGB

    // WGL_ARB_multisample

    // WGL_EXT_swap_control
    PFNWGLSWAPINTERVALEXTPROC               wglSwapIntervalEXT;
    PFNWGLGETSWAPINTERVALEXTPROC            wglGetSwapIntervalEXT;

    // WGL_ARB_pbuffer
    PFNWGLCREATEPBUFFERARBPROC              wglCreatePbufferARB;
    PFNWGLGETPBUFFERDCARBPROC               wglGetPbufferDCARB;
    PFNWGLRELEASEPBUFFERDCARBPROC           wglReleasePbufferDCARB;
    PFNWGLDESTROYPBUFFERARBPROC             wglDestroyPbufferARB;
    PFNWGLQUERYPBUFFERARBPROC               wglQueryPbufferARB;

    bool                                    _swap_control_supported;

public:
    wgl_extensions();

    bool            initialize(std::ostream& os);
    bool            is_initialized() const;
    bool            is_supported(const std::string& ext) const;

private:
    string_set      _wgl_extensions;
    bool            _initialized;

    friend std::ostream& operator<<(std::ostream& out_stream, const wgl_extensions& c);

}; // class wgl_extensions

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm


#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CORE_WM_WIN32_WGL_EXTENSIONS_H_INCLUDED
