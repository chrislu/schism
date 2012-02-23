
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "wgl_extensions.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <ostream>

#include <boost/tokenizer.hpp>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/wm_win32/util/classic_context_win32.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

wgl_extensions::wgl_extensions()
{
    _initialized = false;

    // WGL_ARB_create_context
    wglCreateContextAttribsARB      = 0;
    
    // WGL_ARB_pixel_format
    wglGetPixelFormatAttribivARB    = 0;
    wglGetPixelFormatAttribfvARB    = 0;
    wglChoosePixelFormatARB         = 0;
    
    // WGL_ARB_extensions_string
    wglGetExtensionsStringARB       = 0;

    // WGL_EXT_swap_control
    wglSwapIntervalEXT              = 0;
    wglGetSwapIntervalEXT           = 0;

    // WGL_ARB_pbuffer
    wglCreatePbufferARB             = 0;
    wglGetPbufferDCARB              = 0;
    wglReleasePbufferDCARB          = 0;
    wglDestroyPbufferARB            = 0;
    wglQueryPbufferARB              = 0;

    _swap_control_supported         = false;
}

bool
wgl_extensions::initialize(std::ostream& os)
{
    if (is_initialized()) {
        return (true);
    }
    
    if (0 == wglGetCurrentContext()) {
        os << "wgl_extensions::initialize(): no OpenGL context present, unable to initialize WGL extensions.";
        return (false);
    }

    // WGL_ARB_extensions_string
    wglGetExtensionsStringARB       = (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
    
    if (wglGetExtensionsStringARB == 0) {
        os << "wgl_extensions::initialize(): WGL_ARB_extensions_string not supported";
        return (false);
    }

    // get and tokenize the extension strings
    HDC cur_hdc = wglGetCurrentDC();
    if (cur_hdc == 0) {
        os << "wgl_extensions::initialize(): unable to retrieve current HDC";
        return (false);
    }
    std::string wgl_ext_string = reinterpret_cast<const char*>(wglGetExtensionsStringARB(cur_hdc));

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> space_separator(" ");
    tokenizer                   extension_strings(wgl_ext_string, space_separator);

    for (tokenizer::const_iterator i = extension_strings.begin(); i != extension_strings.end(); ++i) {
        _wgl_extensions.insert(std::string(*i));
    }

    // WGL_ARB_create_context
    wglCreateContextAttribsARB      = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
    
    if (wglCreateContextAttribsARB == 0) {
        os << "wgl_extensions::initialize(): WGL_ARB_create_context not supported";
        return (false);
    }

    // WGL_ARB_pixel_format
    wglGetPixelFormatAttribivARB    = (PFNWGLGETPIXELFORMATATTRIBIVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribivARB");
    wglGetPixelFormatAttribfvARB    = (PFNWGLGETPIXELFORMATATTRIBFVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribfvARB");
    wglChoosePixelFormatARB         = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
    
    if (   wglGetPixelFormatAttribivARB == 0
        || wglGetPixelFormatAttribfvARB == 0
        || wglChoosePixelFormatARB == 0) {
        os << "wgl_extensions::initialize(): WGL_ARB_pixel_format not supported";
        return (false);
    }

    // WGL_EXT_swap_control
    wglSwapIntervalEXT              = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    wglGetSwapIntervalEXT           = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");

    if (   wglSwapIntervalEXT == 0
        || wglGetSwapIntervalEXT == 0) {
        os << "wgl_extensions::initialize(): WGL_EXT_swap_control not supported";
        return (false);
    }

    if (  !is_supported("WGL_ARB_framebuffer_sRGB")
        ||!is_supported("WGL_EXT_framebuffer_sRGB")) {
        os << "wgl_extensions::initialize(): WGL_ARB_framebuffer_sRGB not supported";
    }
    if (!is_supported("WGL_ARB_multisample")) {
        os << "wgl_extensions::initialize(): WGL_ARB_multisample not supported";
    }

    // WGL_ARB_pbuffer
    wglCreatePbufferARB             = (PFNWGLCREATEPBUFFERARBPROC)wglGetProcAddress("wglCreatePbufferARB");
    wglGetPbufferDCARB              = (PFNWGLGETPBUFFERDCARBPROC)wglGetProcAddress("wglGetPbufferDCARB");
    wglReleasePbufferDCARB          = (PFNWGLRELEASEPBUFFERDCARBPROC)wglGetProcAddress("wglReleasePbufferDCARB");
    wglDestroyPbufferARB            = (PFNWGLDESTROYPBUFFERARBPROC)wglGetProcAddress("wglDestroyPbufferARB");
    wglQueryPbufferARB              = (PFNWGLQUERYPBUFFERARBPROC)wglGetProcAddress("wglQueryPbufferARB");

    if (   wglCreatePbufferARB == 0
        || wglGetPbufferDCARB == 0
        || wglReleasePbufferDCARB == 0
        || wglDestroyPbufferARB == 0
        || wglQueryPbufferARB == 0) {
        os << "wgl_extensions::initialize(): WGL_ARB_pbuffer not supported";
        return (false);
    }

    if (is_supported("WGL_EXT_swap_control")) {
        _swap_control_supported = true;
    }

    _initialized = true;

    return (true);
}

bool
wgl_extensions::is_initialized() const
{
    return (_initialized);
}

bool
wgl_extensions::is_supported(const std::string& ext) const
{
    if (_wgl_extensions.find(ext) != _wgl_extensions.end()) {
        return (true);
    }
    else {
        return (false);
    }
}

std::ostream& operator<<(std::ostream& out_stream, const wgl_extensions& w)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "extensions :      " << "(found " << w._wgl_extensions.size() << ")" << std::endl;

    for (wgl_extensions::string_set::const_iterator i = w._wgl_extensions.begin(); i != w._wgl_extensions.end(); ++i) {
        out_stream << "                  " << *i << std::endl;
    }

    return out_stream;
}

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
