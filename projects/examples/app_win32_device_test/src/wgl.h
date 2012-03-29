
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_OPENGL_WGL_H_INCLUDED
#define SCM_GL_UTIL_OPENGL_WGL_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <set>
#include <string>

#include <scm/core/platform/windows.h>

#include <GL/GL.h>
#include <GL/wglext.h>

namespace scm {
namespace gl {
namespace test {

class wgl
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

public:
    wgl();

    bool            initialize();
    bool            is_initialized() const;
    bool            is_supported(const std::string& ext) const;

private:
    string_set      _wgl_extensions;
    bool            _initialized;

}; // class wgl

} // namespace test
} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#endif // SCM_GL_UTIL_OPENGL_WGL_H_INCLUDED
