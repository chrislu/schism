
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "pixel_format_selection_win32.h"

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <sstream>
#include <string>
#include <vector>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/error_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/wgl_extensions.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

int
pixel_format_selector::choose(HDC                               in_device,
                              const surface::format_desc&       in_pfd,
                              const surface_type                in_surface_type,
                              const shared_ptr<wgl_extensions>& in_wgl,
                              std::ostream&                     out_stream)
{
    std::vector<int>  pixel_desc;

    pixel_desc.push_back(WGL_ACCELERATION_ARB);
    pixel_desc.push_back(WGL_FULL_ACCELERATION_ARB);

    if (in_surface_type == window_surface) {
        pixel_desc.push_back(WGL_DRAW_TO_WINDOW_ARB);
        pixel_desc.push_back(GL_TRUE);
    }
    else if (in_surface_type == pbuffer_surface) {
        pixel_desc.push_back(WGL_DRAW_TO_PBUFFER_ARB);
        pixel_desc.push_back(GL_TRUE);
    }
    else {
        out_stream << "pixel_format_selector::choose() <win32>: "
                   << "unknown surface type.";
        return (0);
    }

    pixel_desc.push_back(WGL_SUPPORT_OPENGL_ARB);
    pixel_desc.push_back(GL_TRUE);

    pixel_desc.push_back(WGL_SWAP_METHOD_ARB);
    pixel_desc.push_back(WGL_SWAP_EXCHANGE_ARB);

    // color buffer
    if (is_srgb_format(in_pfd._color_format)) {
        pixel_desc.push_back(WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB);
        pixel_desc.push_back(GL_TRUE);
    }

    pixel_desc.push_back(WGL_COLOR_BITS_ARB);
    pixel_desc.push_back(size_of_format(in_pfd._color_format) * 8);

    pixel_desc.push_back(WGL_ALPHA_BITS_ARB);
    if ((channel_count(in_pfd._color_format) > 3)) {
        pixel_desc.push_back(size_of_channel(in_pfd._color_format) * 8);
    }
    else {
        pixel_desc.push_back(0);
    }

    pixel_desc.push_back(WGL_PIXEL_TYPE_ARB);
    if (is_integer_type(in_pfd._color_format)) {
        pixel_desc.push_back(WGL_TYPE_RGBA_ARB);
    }
    else if (is_packed_format(in_pfd._color_format)) {
        pixel_desc.push_back(WGL_TYPE_RGBA_UNSIGNED_FLOAT_EXT);
    }
    else if (is_float_type(in_pfd._color_format)) {
        pixel_desc.push_back(WGL_TYPE_RGBA_FLOAT_ARB);
    }
    else {
        out_stream << "pixel_format_selector::choose() <win32>: "
                   << "unable to determine pixel type of requested color format "
                   << "(requested: " << format_string(in_pfd._color_format) << ").";
        return (0);
    }
        
    // depth buffer
    pixel_desc.push_back(WGL_DEPTH_BITS_ARB);
    pixel_desc.push_back(size_of_depth_component(in_pfd._depth_stencil_format) * 8);

    pixel_desc.push_back(WGL_STENCIL_BITS_ARB);
    pixel_desc.push_back(size_of_stencil_component(in_pfd._depth_stencil_format) * 8);

    // double, quad buffer
    pixel_desc.push_back(WGL_DOUBLE_BUFFER_ARB);
    pixel_desc.push_back(in_pfd._double_buffer);

    pixel_desc.push_back(WGL_STEREO_ARB);
    pixel_desc.push_back(in_pfd._quad_buffer_stereo);

    pixel_desc.push_back(WGL_SAMPLE_BUFFERS_ARB);   pixel_desc.push_back(GL_FALSE);
    pixel_desc.push_back(WGL_SAMPLES_ARB);          pixel_desc.push_back(0);
    pixel_desc.push_back(WGL_AUX_BUFFERS_ARB);      pixel_desc.push_back(0);

    pixel_desc.push_back(0);                        pixel_desc.push_back(0);

    const int         query_max_formats = 20;
    int               result_pixel_fmts[query_max_formats];
    unsigned int      result_num_pixel_fmts = 0;

    if (in_wgl->wglChoosePixelFormatARB(in_device,
                                        static_cast<const int*>(&(pixel_desc[0])),
                                        NULL,
                                        query_max_formats,
                                        result_pixel_fmts,
                                        &result_num_pixel_fmts) != TRUE)
    {
        out_stream << "pixel_format_selector::choose() <win32>: "
                   << "wglChoosePixelFormat failed - requested pixel format: " << std::endl
                   << in_pfd;
        return (0);
    }

    if (result_num_pixel_fmts < 1) {
        out_stream << "pixel_format_selector::choose() <win32>: "
                   << "wglChoosePixelFormat returned 0 matching pixel formats - requested pixel format: " << std::endl
                   << in_pfd;
        return (0);
    }

    return (result_pixel_fmts[0]);
}

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
