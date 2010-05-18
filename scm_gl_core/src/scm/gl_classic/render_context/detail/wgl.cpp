
#include "wgl.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <boost/tokenizer.hpp>

#include <scm/log.h>
#include <scm/core/pointer_types.h>

#include <scm/gl/graphics_device/opengl3/detail/context_helper.h>

namespace scm {
namespace gl {
namespace detail {

wgl::wgl()
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

}

bool
wgl::initialize()
{
    if (is_initialized()) {
        return (true);
    }

    scm::scoped_ptr<detail::classic_gl_context> dummy_context;
    if (wglGetCurrentContext() == 0) {
        dummy_context.reset(new detail::classic_gl_context);
        if (!dummy_context->create()) {
            scm::err() << scm::log_level(scm::logging::ll_error)
                       << "wgl::initialize(): unable to initialize dummy OpenGL context" << std::endl;
            return (false);
        }
    }

    // WGL_ARB_extensions_string
    wglGetExtensionsStringARB       = (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
    
    if (wglGetExtensionsStringARB == 0) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "wgl::initialize(): WGL_ARB_extensions_string not supported" << std::endl;
        return (false);
    }

    // get and tokenize the extension strings
    HDC cur_hdc = wglGetCurrentDC();
    if (cur_hdc == 0) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "wgl::initialize(): unable to retrieve current HDC" << std::endl;
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
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "wgl::initialize(): WGL_ARB_create_context not supported" << std::endl;
        return (false);
    }

    // WGL_ARB_pixel_format
    wglGetPixelFormatAttribivARB    = (PFNWGLGETPIXELFORMATATTRIBIVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribivARB");
    wglGetPixelFormatAttribfvARB    = (PFNWGLGETPIXELFORMATATTRIBFVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribfvARB");
    wglChoosePixelFormatARB         = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
    
    if (   wglGetPixelFormatAttribivARB == 0
        || wglGetPixelFormatAttribfvARB == 0
        || wglChoosePixelFormatARB == 0) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "wgl::initialize(): WGL_ARB_pixel_format not supported" << std::endl;
        return (false);
    }

    // WGL_EXT_swap_control
    wglSwapIntervalEXT              = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    wglGetSwapIntervalEXT           = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");

    if (   wglSwapIntervalEXT == 0
        || wglGetSwapIntervalEXT == 0) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "wgl::initialize(): WGL_EXT_swap_control not supported" << std::endl;
        return (false);
    }

    if (!is_supported("WGL_ARB_framebuffer_sRGB")) {
        scm::out() << scm::log_level(scm::logging::ll_warning)
                   << "wgl::initialize(): WGL_ARB_framebuffer_sRGB not supported" << std::endl;
    }
    if (!is_supported("WGL_ARB_multisample")) {
        scm::out() << scm::log_level(scm::logging::ll_warning)
                   << "wgl::initialize(): WGL_ARB_multisample not supported" << std::endl;
    }

    _initialized = true;

    return (true);
}

bool
wgl::is_initialized() const
{
    return (_initialized);
}

bool
wgl::is_supported(const std::string& ext) const
{
    if (_wgl_extensions.find(ext) != _wgl_extensions.end()) {
        return (true);
    }
    else {
        return (false);
    }
}

} //namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
