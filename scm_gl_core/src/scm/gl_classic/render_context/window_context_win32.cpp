
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "window_context_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <GL/glew.h>
#include <GL/wglew.h>

#include <string>
#include <vector>

#include <boost/bind.hpp>

#include <scm/log.h>

// HACK FUCKING GLEW
#define WGL_CONTEXT_PROFILE_MASK_ARB		        0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB	        0x00000001
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB   0x00000002

#define ERROR_INVALID_VERSION_ARB                   0x2095
#define ERROR_INVALID_PROFILE_ARB                   0x2096

namespace scm {
namespace gl_classic {

namespace detail {

void null_deleter(context_base::handle::element_type* p)
{
}

class old_school_gl_context
{
public:
    old_school_gl_context() : _hWnd(NULL), _hDC(NULL), _hGLRC(NULL), _wnd_class_name("gl_dummy_for_wgl_stuff") {}
    virtual ~old_school_gl_context()
    {
        if (_hGLRC != NULL) {
            destroy();
        }
    }
    bool create(const context_format& desc)
    {
        if (_hGLRC != NULL) {
            destroy();
        }
        // Register A Window Class
        WNDCLASSEX wnd_class;
        ZeroMemory (&wnd_class, sizeof (WNDCLASSEX));
        wnd_class.cbSize			= sizeof (WNDCLASSEX);
        wnd_class.lpfnWndProc       = (WNDPROC)DefWindowProc;
    	wnd_class.style			    = CS_OWNDC;
        wnd_class.lpszClassName	    = _wnd_class_name.c_str();
        wnd_class.hInstance         = GetModuleHandle(NULL);

        if (RegisterClassEx(&wnd_class) == 0)
        {
	        return (false);
        }

        // window
        _hWnd = CreateWindow(  _wnd_class_name.c_str(),
							   _wnd_class_name.c_str(),
							   WS_POPUP,
							   0, 0,
							   10,
							   10,
							   NULL,
							   NULL,
							   GetModuleHandle(NULL),
							   NULL);

        if (_hWnd == 0)
        {
	        return (false);	
        }

        _hDC = GetDC(_hWnd);

        if (_hDC == NULL) {
            destroy();
            return (false);
        }
        DWORD dwflags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER_DONTCARE;

#if SCM_WIN_VER >= SCM_WIN_VER_VISTA
        dwflags |= PFD_SUPPORT_COMPOSITION;
#endif
        PIXELFORMATDESCRIPTOR pfd =  {
            sizeof(PIXELFORMATDESCRIPTOR),
            1,
            dwflags,
            PFD_TYPE_RGBA,
            desc.color_bits(),      // color bits
            0, 0, 0, 0, 0, 0, 0, 0, // color/alpha bits/shift
            0,                      // accum bits
            0, 0, 0, 0,             // accum rgba bits
            desc.depth_bits(),
            desc.stencil_bits(),
            0,                      // aux buffers
            PFD_MAIN_PLANE,
            0,
            0, 0, 0
        };

        int pfmt = ChoosePixelFormat(_hDC, &pfd);

        if (pfmt < 1) {
            destroy();
            return (false);
        }

        if (!SetPixelFormat(_hDC, pfmt, &pfd)) {
            destroy();
            return (false);
        }

        _hGLRC = wglCreateContext(_hDC);

        if (_hGLRC == NULL) {
            destroy();
            return (false);
        }

        wglMakeCurrent(_hDC,_hGLRC);

        return (true);
    }
    void destroy()
    {
        if (_hGLRC != NULL) {
            wglMakeCurrent(_hDC, NULL);
            wglDeleteContext(_hGLRC);
            _hGLRC = NULL;
        }

        if (_hDC != NULL) {
            ReleaseDC(_hWnd, _hDC);
            _hDC = NULL;
        }

        if (_hWnd != NULL) {
            DestroyWindow(_hWnd);
            UnregisterClass(_wnd_class_name.c_str(), GetModuleHandle(NULL));
            _hWnd = NULL;
        }
    }
private:
    HWND    _hWnd;
    HDC     _hDC;
    HGLRC   _hGLRC;

    std::string _wnd_class_name;
};

} // namespace detail

window_context_win32::window_context_win32()
  : _swap_control_supported(false)
{
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)									// Check For Windows Messages
    {
    case WM_SYSCOMMAND:							// Intercept System Commands
    {
        switch (wParam)							// Check System Calls
        {
        case SC_SCREENSAVE:					// Screensaver Trying To Start?
        case SC_MONITORPOWER:				// Monitor Trying To Enter Powersave?
            return 0;							// Prevent From Happening
        }
        break;									// Exit
    }
    case WM_ERASEBKGND:
    {
        return 0;
    }break;

    }

    // Pass All Unhandled Messages To DefWindowProc
    return DefWindowProc(hWnd,uMsg,wParam,lParam);
}

window_context_win32::~window_context_win32()
{
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

    //LONG dwexstyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;;
    //LONG dwstyle   = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;

    //if (SetWindowLong(static_cast<HWND>(_wnd_handle.get()), GWL_EXSTYLE, dwexstyle) == 0) {
    //    scm::err() << log::warning
    //               << "context_win32::set_up(): "
    //               << "SetWindowLong failed for GWL_EXSTYLE: "
    //               << std::hex << dwexstyle << log::end;
    //}
    //if (SetWindowLong(static_cast<HWND>(_wnd_handle.get()), GWL_STYLE, dwstyle) == 0) {
    //    scm::err() << log::warning
    //               << "context_win32::set_up(): "
    //               << "SetWindowLong failed for GWL_STYLE: "
    //               << std::hex << dwstyle << log::end;
    //}
    //if (SetWindowLongPtr(static_cast<HWND>(_wnd_handle.get()), GWLP_WNDPROC, (LONG_PTR)&WndProc) == 0) {
    //    scm::err() << log::warning
    //               << "context_win32::set_up(): "
    //               << "SetWindowLong failed for GWL_WNDPROC: "
    //               << std::hex << &WndProc << log::end;
    //}

    _device_handle.reset(GetDC(static_cast<HWND>(_wnd_handle.get())),
                               boost::bind<int>(ReleaseDC, static_cast<HWND>(_wnd_handle.get()), _1));

    if (!_device_handle) {
        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "unable to retrive device context (GetDC failed on window handle: "
                   << std::hex << _wnd_handle.get() << ")" << log::end;
        return (false);
    }

    detail::old_school_gl_context dummy_context;

    if (!dummy_context.create(desc)) {
        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "unable setup dummy gl context to initialize WGL ARB extensions" << log::end;
        return (false);
    }

    if (glewInit() != GLEW_OK) {
        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "unable to initialize WGL ARB extensions, glewInit failed" << log::end;
        dummy_context.destroy();
        return (false);
    }
    // we do not need you anymore
    dummy_context.destroy();

    std::vector<int>  pixel_desc;

    pixel_desc.push_back(WGL_ACCELERATION_ARB);     pixel_desc.push_back(WGL_FULL_ACCELERATION_ARB);
    pixel_desc.push_back(WGL_DRAW_TO_WINDOW_ARB);   pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(WGL_SUPPORT_OPENGL_ARB);   pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(WGL_SWAP_METHOD_ARB);      pixel_desc.push_back(WGL_SWAP_EXCHANGE_ARB);
    pixel_desc.push_back(WGL_PIXEL_TYPE_ARB);       pixel_desc.push_back(WGL_TYPE_RGBA_ARB);

    pixel_desc.push_back(WGL_DOUBLE_BUFFER_ARB);    pixel_desc.push_back(desc.double_buffer());
    pixel_desc.push_back(WGL_STEREO_ARB);           pixel_desc.push_back(desc.stereo());
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

    if (wglChoosePixelFormatARB(static_cast<HDC>(_device_handle.get()),
                                static_cast<const int*>(&(pixel_desc[0])),
                                NULL,
                                query_max_formats,
                                result_pixel_fmts,
                                &result_num_pixel_fmts) != TRUE)
    {
        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "wglChoosePixelFormat failed" << log::end;
        return (false);
    }

    if (result_num_pixel_fmts < 1) {
        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "wglChoosePixelFormat returned 0 matching pixel formats" << log::end;
        return (false);
    }

    if (SetPixelFormat(static_cast<HDC>(_device_handle.get()), result_pixel_fmts[0], NULL) != TRUE) {
        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "SetPixelFormat failed for format number: " << result_pixel_fmts[0] << log::end;
        return (false);
    }

    if (!wglewIsSupported("WGL_ARB_create_context")){
        scm::err() << log::warning
                   << "context_win32::set_up(): "
                   << "WGL_ARB_create_context not supported: "
                   << "using default wglCreateContest function which does not allow request of versioned OpenGL context" << log::end;

        _context_handle.reset(wglCreateContext(static_cast<HDC>(_device_handle.get())),
                              boost::bind<BOOL>(wglDeleteContext, _1));

        if (!share_ctx.empty()/* != empty_context()*/) {
            if (wglShareLists(static_cast<HGLRC>(share_ctx.context_handle().get()),
                              static_cast<HGLRC>(_context_handle.get())) == FALSE) {

                scm::err() << log::error
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
        if (wglewIsSupported("WGL_ARB_create_context_profile")) {
            if (desc.compatibility_profile()) {
                ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB);
            }
            else {
                ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_CORE_PROFILE_BIT_ARB);
            }
        }
		}
        ctx_attribs.push_back(0);                                   ctx_attribs.push_back(0); // terminate list

        _context_handle.reset(wglCreateContextAttribsARB(static_cast<HDC>(_device_handle.get()),
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

        scm::err() << log::error
                   << "context_win32::set_up(): "
                   << "unable to create OpenGL context (wglCreateContextAttribsARB failed [" << es << "])"  << log::end;
        return (false);
    }

    if (wglewIsSupported("WGL_EXT_swap_control")) {
        _swap_control_supported = true;
    }
    else {
        scm::err() << log::warning
                   << "context_win32::set_up(): "
                   << "WGL_EXT_swap_control not supported, operating system default behavior used."  << log::end;
    }

    _context_format = desc;
    make_current(true);

    std::string wgl_ext_string = (char*)wglGetExtensionsStringARB(static_cast<HDC>(_device_handle.get()));
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
    if (_swap_control_supported) {
        wglSwapIntervalEXT(interval);
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

} // namespace gl_classic
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
