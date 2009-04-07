
#include "device_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <vector>

#include <boost/bind.hpp>

#include <scm/log.h>
#include <scm/core/platform/windows.h>
#include <GL/gl.h>

#include <scm/gl/graphics_device/opengl3/detail/wgl.h>

namespace scm {
namespace gl {

namespace detail {
} // namespace detail

opengl_device_win32::opengl_device_win32(const device_initializer& init,
                                         const device_context_config& cfg)
  : opengl_device(init, cfg)
{
    _wgl.reset(new detail::wgl());
    if (!_wgl->initialize()) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device_win32::setup_render_context(): "
                   << "wglChoosePixelFormat failed" << std::endl;
        throw (std::runtime_error("opengl_device::opengl_device(): error initializing WGL"));
    }

    if (   cfg._context_type != device_context_config::CONTEXT_WINDOWED
        || cfg._context_type != device_context_config::CONTEXT_FULLSCREEN) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device_win32::setup_render_context(): "
                   << "only windowed and fullscreen contexts supported as main context" << std::endl;
        throw (std::runtime_error("opengl_device::opengl_device(): error"));
    }

    // check if opengl is available

    // check if requested feature level is supported
    // if not throw runtime_error or something
}

opengl_device_win32::~opengl_device_win32()
{
}

bool
opengl_device_win32::setup_render_context(const device_context_config& cfg, unsigned feature_level)
{
    _hDC.reset(GetDC(static_cast<HWND>(cfg._output_window.get())),
               boost::bind<int>(ReleaseDC, static_cast<HWND>(cfg._output_window.get()), _1));

    if (!_hDC) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device::setup_render_context(): "
                   << "unable to retrive device context (GetDC failed on window handle: "
                   << std::hex << cfg._output_window.get() << ")" << std::endl;
        return (false);
    }

    std::vector<int>  pixel_desc;

    using detail::wgl;

    pixel_desc.push_back(wgl::WGL_ACCELERATION_ARB);         pixel_desc.push_back(wgl::WGL_FULL_ACCELERATION_ARB);
    pixel_desc.push_back(wgl::WGL_DRAW_TO_WINDOW_ARB);       pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(wgl::WGL_SUPPORT_OPENGL_ARB);       pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(wgl::WGL_SWAP_METHOD_ARB);          pixel_desc.push_back(wgl::WGL_SWAP_EXCHANGE_ARB);
    pixel_desc.push_back(wgl::WGL_PIXEL_TYPE_ARB);           pixel_desc.push_back(wgl::WGL_TYPE_RGBA_ARB);

    pixel_desc.push_back(wgl::WGL_DOUBLE_BUFFER_ARB);        pixel_desc.push_back((cfg._color_buffer_count > 1) ? GL_TRUE : GL_FALSE);
    //pixel_desc.push_back(WGL_STEREO_ARB);                    pixel_desc.push_back(desc.stereo());
    pixel_desc.push_back(wgl::WGL_SAMPLE_BUFFERS_ARB);       pixel_desc.push_back(cfg._sample_count > 1 ? GL_TRUE : GL_FALSE);
    pixel_desc.push_back(wgl::WGL_SAMPLES_ARB);              pixel_desc.push_back(cfg._sample_count);

    switch (cfg._color_buffer_format) {
        case FORMAT_RGB8: {
            pixel_desc.push_back(wgl::WGL_COLOR_BITS_ARB);   pixel_desc.push_back(32);
            pixel_desc.push_back(wgl::WGL_ALPHA_BITS_ARB);   pixel_desc.push_back(0);
        } break;
        case FORMAT_RGBA8: {
            pixel_desc.push_back(wgl::WGL_COLOR_BITS_ARB);   pixel_desc.push_back(32);
            pixel_desc.push_back(wgl::WGL_ALPHA_BITS_ARB);   pixel_desc.push_back(8);
        } break;
    }
    switch (cfg._depth_stencil_buffer_format) {
        case FORMAT_D24: {
            pixel_desc.push_back(wgl::WGL_DEPTH_BITS_ARB);   pixel_desc.push_back(24);
            pixel_desc.push_back(wgl::WGL_STENCIL_BITS_ARB); pixel_desc.push_back(0);
        } break;
        case FORMAT_D24_S8: {
            pixel_desc.push_back(wgl::WGL_DEPTH_BITS_ARB);   pixel_desc.push_back(24);
            pixel_desc.push_back(wgl::WGL_STENCIL_BITS_ARB); pixel_desc.push_back(8);
        } break;
    }

    pixel_desc.push_back(0);                        pixel_desc.push_back(0); // terminate list

    const int         query_max_formats = 20;
    int               result_pixel_fmts[query_max_formats];
    unsigned int      result_num_pixel_fmts = 0;

    if (_wgl->wglChoosePixelFormatARB(static_cast<HDC>(_hDC.get()),
                                      static_cast<const int*>(&(pixel_desc[0])),
                                      0,
                                      query_max_formats,
                                      result_pixel_fmts,
                                      &result_num_pixel_fmts) != TRUE)
    {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device::setup_render_context(): "
                   << "wglChoosePixelFormat failed" << std::endl;
        return (false);
    }

    if (result_num_pixel_fmts < 1) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device::setup_render_context(): "
                   << "wglChoosePixelFormat returned 0 matching pixel formats" << std::endl;
        return (false);
    }

    if (SetPixelFormat(static_cast<HDC>(_hDC.get()), result_pixel_fmts[0], NULL) != TRUE) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device::setup_render_context(): "
                   << "SetPixelFormat failed for format number: " << result_pixel_fmts[0] << std::endl;
        return (false);
    }

    if (feature_level < opengl_device::OPENGL_FEATURE_LEVEL_3_0) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "opengl_device::setup_render_context(): "
                   << "currently only feature levels > OpenGL 3.0 supported" << std::endl;
        return (false);
    }

    std::vector<int>  ctx_attribs;

    switch (feature_level) {
        case opengl_device::OPENGL_FEATURE_LEVEL_2_1: {
            ctx_attribs.push_back(wgl::WGL_CONTEXT_MAJOR_VERSION_ARB);      ctx_attribs.push_back(2);
            ctx_attribs.push_back(wgl::WGL_CONTEXT_MINOR_VERSION_ARB);      ctx_attribs.push_back(1);
        } break;
        case opengl_device::OPENGL_FEATURE_LEVEL_3_0: {
            ctx_attribs.push_back(wgl::WGL_CONTEXT_MAJOR_VERSION_ARB);      ctx_attribs.push_back(3);
            ctx_attribs.push_back(wgl::WGL_CONTEXT_MINOR_VERSION_ARB);      ctx_attribs.push_back(0);
            ctx_attribs.push_back(wgl::WGL_CONTEXT_FLAGS_ARB);              ctx_attribs.push_back(wgl::WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
        } break;
        case opengl_device::OPENGL_FEATURE_LEVEL_3_1: {
            ctx_attribs.push_back(wgl::WGL_CONTEXT_MAJOR_VERSION_ARB);      ctx_attribs.push_back(3);
            ctx_attribs.push_back(wgl::WGL_CONTEXT_MINOR_VERSION_ARB);      ctx_attribs.push_back(1);
            ctx_attribs.push_back(wgl::WGL_CONTEXT_FLAGS_ARB);              ctx_attribs.push_back(wgl::WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
        } break;
    };
    //if (desc.debug()) {
    //    ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(WGL_CONTEXT_DEBUG_BIT_ARB);
    //}
    ctx_attribs.push_back(0);                                               ctx_attribs.push_back(0); // terminate list

    //_context_handle.reset(_wgl->wglCreateContextAttribsARB(static_cast<HDC>(_hDC.get()),
    //                                                       static_cast<HGLRC>(share_ctx.context_handle().get()),
    //                                                       static_cast<const int*>(&(ctx_attribs[0]))),
    //                      boost::bind<BOOL>(wglDeleteContext, _1));

    //if (!_context_handle) {
    //    scm::err() << scm::log_level(scm::logging::ll_error)
    //               << "opengl_device::setup_render_context(): "
    //               << "SetPixelFormat failed for format number: " << result_pixel_fmts[0] << std::endl;

    //    return (false);
    //}

    //_context_format = desc;
    //make_current(true);

    return (false);
}

} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
