
#ifndef SCM_GL_DETAIL_GL_WGL_H_INCLUDED
#define SCM_GL_DETAIL_GL_WGL_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <set>
#include <string>


#include <scm/core/platform/windows.h>

namespace scm {
namespace gl {
namespace detail {

class wgl
{
private:
    typedef std::set<std::string> string_set;
    // WGL_ARB_create_context
    typedef HGLRC (WINAPI * PFNWGLCREATECONTEXTATTRIBSARBPROC) (HDC hDC, HGLRC hShareContext, const int *attribList);

    // WGL_ARB_pixel_format
    typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, int *piValues);
    typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, FLOAT *pfValues);
    typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);

    // WGL_ARB_extensions_string
    typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);

    // WGL_EXT_swap_control
    typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
    typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);

public:
    // WGL_ARB_create_context
    static const int WGL_CONTEXT_DEBUG_BIT_ARB              = 0x0001;
    static const int WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB = 0x0002;
    static const int WGL_CONTEXT_MAJOR_VERSION_ARB          = 0x2091;
    static const int WGL_CONTEXT_MINOR_VERSION_ARB          = 0x2092;
    static const int WGL_CONTEXT_LAYER_PLANE_ARB            = 0x2093;
    static const int WGL_CONTEXT_FLAGS_ARB                  = 0x2094;
    static const int ERROR_INVALID_VERSION_ARB              = 0x2095;

    PFNWGLCREATECONTEXTATTRIBSARBPROC       wglCreateContextAttribsARB;
    
    // WGL_ARB_pixel_format
    static const int WGL_NUMBER_PIXEL_FORMATS_ARB           = 0x2000;
    static const int WGL_DRAW_TO_WINDOW_ARB                 = 0x2001;
    static const int WGL_DRAW_TO_BITMAP_ARB                 = 0x2002;
    static const int WGL_ACCELERATION_ARB                   = 0x2003;
    static const int WGL_NEED_PALETTE_ARB                   = 0x2004;
    static const int WGL_NEED_SYSTEM_PALETTE_ARB            = 0x2005;
    static const int WGL_SWAP_LAYER_BUFFERS_ARB             = 0x2006;
    static const int WGL_SWAP_METHOD_ARB                    = 0x2007;
    static const int WGL_NUMBER_OVERLAYS_ARB                = 0x2008;
    static const int WGL_NUMBER_UNDERLAYS_ARB               = 0x2009;
    static const int WGL_TRANSPARENT_ARB                    = 0x200A;
    static const int WGL_TRANSPARENT_RED_VALUE_ARB          = 0x2037;
    static const int WGL_TRANSPARENT_GREEN_VALUE_ARB        = 0x2038;
    static const int WGL_TRANSPARENT_BLUE_VALUE_ARB         = 0x2039;
    static const int WGL_TRANSPARENT_ALPHA_VALUE_ARB        = 0x203A;
    static const int WGL_TRANSPARENT_INDEX_VALUE_ARB        = 0x203B;
    static const int WGL_SHARE_DEPTH_ARB                    = 0x200C;
    static const int WGL_SHARE_STENCIL_ARB                  = 0x200D;
    static const int WGL_SHARE_ACCUM_ARB                    = 0x200E;
    static const int WGL_SUPPORT_GDI_ARB                    = 0x200F;
    static const int WGL_SUPPORT_OPENGL_ARB                 = 0x2010;
    static const int WGL_DOUBLE_BUFFER_ARB                  = 0x2011;
    static const int WGL_STEREO_ARB                         = 0x2012;
    static const int WGL_PIXEL_TYPE_ARB                     = 0x2013;
    static const int WGL_COLOR_BITS_ARB                     = 0x2014;
    static const int WGL_RED_BITS_ARB                       = 0x2015;
    static const int WGL_RED_SHIFT_ARB                      = 0x2016;
    static const int WGL_GREEN_BITS_ARB                     = 0x2017;
    static const int WGL_GREEN_SHIFT_ARB                    = 0x2018;
    static const int WGL_BLUE_BITS_ARB                      = 0x2019;
    static const int WGL_BLUE_SHIFT_ARB                     = 0x201A;
    static const int WGL_ALPHA_BITS_ARB                     = 0x201B;
    static const int WGL_ALPHA_SHIFT_ARB                    = 0x201C;
    static const int WGL_ACCUM_BITS_ARB                     = 0x201D;
    static const int WGL_ACCUM_RED_BITS_ARB                 = 0x201E;
    static const int WGL_ACCUM_GREEN_BITS_ARB               = 0x201F;
    static const int WGL_ACCUM_BLUE_BITS_ARB                = 0x2020;
    static const int WGL_ACCUM_ALPHA_BITS_ARB               = 0x2021;
    static const int WGL_DEPTH_BITS_ARB                     = 0x2022;
    static const int WGL_STENCIL_BITS_ARB                   = 0x2023;
    static const int WGL_AUX_BUFFERS_ARB                    = 0x2024;
    static const int WGL_NO_ACCELERATION_ARB                = 0x2025;
    static const int WGL_GENERIC_ACCELERATION_ARB           = 0x2026;
    static const int WGL_FULL_ACCELERATION_ARB              = 0x2027;
    static const int WGL_SWAP_EXCHANGE_ARB                  = 0x2028;
    static const int WGL_SWAP_COPY_ARB                      = 0x2029;
    static const int WGL_SWAP_UNDEFINED_ARB                 = 0x202A;
    static const int WGL_TYPE_RGBA_ARB                      = 0x202B;
    static const int WGL_TYPE_COLORINDEX_ARB                = 0x202C;

    PFNWGLGETPIXELFORMATATTRIBIVARBPROC     wglGetPixelFormatAttribivARB;
    PFNWGLGETPIXELFORMATATTRIBFVARBPROC     wglGetPixelFormatAttribfvARB;
    PFNWGLCHOOSEPIXELFORMATARBPROC          wglChoosePixelFormatARB;

    // WGL_ARB_extensions_string
    PFNWGLGETEXTENSIONSSTRINGARBPROC        wglGetExtensionsStringARB;

    // ARB_framebuffer_sRGB
    static const int WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB       = 0x20A9;

    // WGL_ARB_multisample
    static const int WGL_SAMPLE_BUFFERS_ARB                 = 0x2041;
    static const int WGL_SAMPLES_ARB                        = 0x2042;

    // WGL_EXT_swap_control
    PFNWGLSWAPINTERVALEXTPROC               wglSwapIntervalEXT;
    PFNWGLGETSWAPINTERVALEXTPROC            wglGetSwapIntervalEXT;

public:
    wgl();

    bool            initialize();
    bool            is_initialized() const;
    bool            is_supported(const std::string& ext) const;

private:
    string_set      _wgl_extensions;
    bool            _initialized;

}; // class wgl

} // namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#endif // SCM_GL_DETAIL_GL_WGL_WGL_H_INCLUDED
