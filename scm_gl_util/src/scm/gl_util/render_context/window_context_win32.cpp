
#include "window_context_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>
#include <scm/gl_util/render_context/GL/gl3.h>
#include <scm/gl_util/render_context/GL/wglext.h>

#include <string>
#include <vector>

#include <boost/bind.hpp>

#include <scm/gl_core/log.h>

namespace scm {
namespace gl {

namespace detail {

void null_deleter(context_base::handle::element_type* p)
{
}

} // namespace detail

window_context_win32::window_context_win32()
  : _swap_control_supported(false),
    _wgl(new detail::wgl())
{
}

window_context_win32::~window_context_win32()
{
    _wgl.reset();
    cleanup();
}

bool
window_context_win32::setup(const wnd_handle hwnd,
                            const context_format& desc)
{
    return (setup(hwnd, desc, empty_context()));
}

bool
window_context_win32::setup(const wnd_handle hwnd,
                            const context_format& desc,
                            const context_base& share_ctx)
{
    _wnd_handle.reset(hwnd, detail::null_deleter);

    _device_handle.reset(GetDC(static_cast<HWND>(_wnd_handle.get())),
                               boost::bind<int>(ReleaseDC, static_cast<HWND>(_wnd_handle.get()), _1));

    if (!_device_handle) {
        glerr() << log::error
                << "context_win32::set_up(): "
                << "unable to retrive device context (GetDC failed on window handle: "
                << std::hex << _wnd_handle.get() << ")" << log::end;
        return (false);
    }



    if (!_wgl->initialize()) {
        glerr() << log::error
                << "context_win32::set_up(): "
                << "unable to initialize WGL ARB extensions, WGL init failed" << log::end;
        return (false);
    }

    std::vector<int>  pixel_desc;

    pixel_desc.push_back(WGL_ACCELERATION_ARB);     pixel_desc.push_back(WGL_FULL_ACCELERATION_ARB);
    pixel_desc.push_back(WGL_DRAW_TO_WINDOW_ARB);   pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(WGL_SUPPORT_OPENGL_ARB);   pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(WGL_SWAP_METHOD_ARB);      pixel_desc.push_back(WGL_SWAP_EXCHANGE_ARB);
    pixel_desc.push_back(WGL_PIXEL_TYPE_ARB);       pixel_desc.push_back(WGL_TYPE_RGBA_ARB);

    pixel_desc.push_back(WGL_DOUBLE_BUFFER_ARB);    pixel_desc.push_back(desc.double_buffer());
    pixel_desc.push_back(WGL_STEREO_ARB);           pixel_desc.push_back(desc.quad_buffer_stereo());
    pixel_desc.push_back(WGL_COLOR_BITS_ARB);       pixel_desc.push_back(desc.color_bits());
    pixel_desc.push_back(WGL_ALPHA_BITS_ARB);       pixel_desc.push_back(desc.alpha_bits());
    pixel_desc.push_back(WGL_DEPTH_BITS_ARB);       pixel_desc.push_back(desc.depth_bits());
    pixel_desc.push_back(WGL_STENCIL_BITS_ARB);     pixel_desc.push_back(desc.stencil_bits());
    pixel_desc.push_back(WGL_SAMPLE_BUFFERS_ARB);   pixel_desc.push_back(desc.max_samples() > 0 ? GL_TRUE : GL_FALSE);
    pixel_desc.push_back(WGL_SAMPLES_ARB);          pixel_desc.push_back(desc.max_samples());
    pixel_desc.push_back(WGL_AUX_BUFFERS_ARB);      pixel_desc.push_back(desc.max_aux_buffers());

    pixel_desc.push_back(0);                        pixel_desc.push_back(0); // terminate list

    const int         query_max_formats = 20;
    int               result_pixel_fmts[query_max_formats];
    unsigned int      result_num_pixel_fmts = 0;

    if (_wgl->wglChoosePixelFormatARB(static_cast<HDC>(_device_handle.get()),
                                      static_cast<const int*>(&(pixel_desc[0])),
                                      NULL,
                                      query_max_formats,
                                      result_pixel_fmts,
                                      &result_num_pixel_fmts) != TRUE)
    {
        glerr() << log::error
                << "context_win32::set_up(): "
                << "wglChoosePixelFormat failed" << log::end;
        return (false);
    }

    if (result_num_pixel_fmts < 1) {
        glerr() << log::error
                << "context_win32::set_up(): "
                << "wglChoosePixelFormat returned 0 matching pixel formats" << log::end;
        return (false);
    }

    if (SetPixelFormat(static_cast<HDC>(_device_handle.get()), result_pixel_fmts[0], NULL) != TRUE) {
        glerr() << log::error
                << "context_win32::set_up(): "
                << "SetPixelFormat failed for format number: " << result_pixel_fmts[0] << log::end;
        return (false);
    }

    if (!_wgl->is_supported("WGL_ARB_create_context")){
        glerr() << log::warning
                << "context_win32::set_up(): "
                << "WGL_ARB_create_context not supported: "
                << "using default wglCreateContest function which does not allow request of versioned OpenGL context" << log::end;

        _context_handle.reset(wglCreateContext(static_cast<HDC>(_device_handle.get())),
                              boost::bind<BOOL>(wglDeleteContext, _1));

        if (!share_ctx.empty()/* != empty_context()*/) {
            if (wglShareLists(static_cast<HGLRC>(share_ctx.context_handle().get()),
                              static_cast<HGLRC>(_context_handle.get())) == FALSE) {

                glerr() << log::error
                        << "context_win32::set_up(): "
                        << "wglShareLists failed (this: " << std::hex << _context_handle.get()
                        << ", share: " << std::hex << share_ctx.context_handle().get() << log::end;
                return (false);
            }
        }
    }
    else {
        std::vector<int>  ctx_attribs;
		if(desc.version_major() > 2) {
            ctx_attribs.push_back(WGL_CONTEXT_MAJOR_VERSION_ARB);       ctx_attribs.push_back(desc.version_major());
            ctx_attribs.push_back(WGL_CONTEXT_MINOR_VERSION_ARB);       ctx_attribs.push_back(desc.version_minor());
            if (desc.forward_compatible()) {
                ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
            }
            if (desc.debug()) {
                ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(WGL_CONTEXT_DEBUG_BIT_ARB);
            }
            if (_wgl->is_supported("WGL_ARB_create_context_profile")) {
                if (desc.compatibility_profile()) {
                    ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB);
                }
                else {
                    ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_CORE_PROFILE_BIT_ARB);
                }
            }
		}
        ctx_attribs.push_back(0);                                   ctx_attribs.push_back(0); // terminate list

        _context_handle.reset(_wgl->wglCreateContextAttribsARB(static_cast<HDC>(_device_handle.get()),
                                                               static_cast<HGLRC>(share_ctx.context_handle().get()),
                                                               static_cast<const int*>(&(ctx_attribs[0]))),
                              boost::bind<BOOL>(wglDeleteContext, _1));
    }

    if (!_context_handle) {
        DWORD           e = GetLastError();
        std::string     es;

        switch (e) {
            case ERROR_INVALID_VERSION_ARB: es.assign("ERROR_INVALID_VERSION_ARB");break;
            case ERROR_INVALID_PROFILE_ARB: es.assign("ERROR_INVALID_PROFILE_ARB");break;
            default: es.assign("unknown error");
        }

        glerr() << log::error
                << "context_win32::set_up(): "
                << "unable to create OpenGL context (wglCreateContextAttribsARB failed [" << es << "])"  << log::end;
        return (false);
    }

    if (_wgl->is_supported("WGL_EXT_swap_control")) {
        _swap_control_supported = true;
    }
    else {
        glerr() << log::warning
                << "context_win32::set_up(): "
                << "WGL_EXT_swap_control not supported, operating system default behavior used."  << log::end;
    }

    _context_format = desc;
    make_current(true);

    std::string wgl_ext_string = _wgl->wglGetExtensionsStringARB(static_cast<HDC>(_device_handle.get()));
    std::replace(wgl_ext_string.begin(), wgl_ext_string.end(), ' ', '\n');

    std::cout << "OpenGL WGL extensions: " << std::endl;
    std::cout << wgl_ext_string << std::endl;


    return (true);
}

bool
window_context_win32::make_current(bool current) const
{
    return (wglMakeCurrent(static_cast<HDC>(_device_handle.get()),
                           current ? static_cast<HGLRC>(_context_handle.get()) : NULL) == TRUE ? true : false);
}

void window_context_win32::swap_buffers(int interval) const
{
    assert(_wgl->is_initialized());

    if (_swap_control_supported) {
        _wgl->wglSwapIntervalEXT(interval);
    }
    SwapBuffers(static_cast<HDC>(_device_handle.get()));
}

void window_context_win32::cleanup()
{
    make_current(false);
    _context_handle.reset();
    _device_handle.reset();
    //wglDeleteContext(this->_hGLRC);
    //ReleaseDC(_wnd_handle, this->_hDC);
}

/*static*/
window_context_win32&
window_context_win32::empty_context()
{
    static window_context_win32 emptyctx;
    return (emptyctx);
}

const shared_ptr<detail::wgl>
window_context_win32::wgl() const
{
    return (_wgl);
}

} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
