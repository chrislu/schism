
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "classic_context_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <GL/gl.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {


classic_gl_window::classic_gl_window(const math::vec2i&       in_position,
                                     const math::vec2ui&      in_size)
  : _hWnd(NULL),
    _hDC(NULL),
    _wnd_class_name("gl_dummy_for_wgl_stuff")
{
    // Register A Window Class
    WNDCLASSEX wnd_class;
    ZeroMemory (&wnd_class, sizeof (WNDCLASSEX));
    wnd_class.cbSize			= sizeof (WNDCLASSEX);
    wnd_class.lpfnWndProc       = (WNDPROC)DefWindowProc;
	wnd_class.style			    = CS_OWNDC;
    wnd_class.lpszClassName	    = _wnd_class_name.c_str();
    wnd_class.hInstance         = ::GetModuleHandle(NULL);

    if (::RegisterClassEx(&wnd_class) == 0) {
        destroy();
    }

    // window
    _hWnd = ::CreateWindow(_wnd_class_name.c_str(),
						   _wnd_class_name.c_str(),
						   WS_POPUP | WS_DISABLED,
						   in_position.x, in_position.y,
						   in_size.x, in_size.y,
						   NULL,
						   NULL,
						   ::GetModuleHandle(NULL),
						   NULL);

    if (0 == _hWnd) {
        destroy();
    }

    _hDC = ::GetDC(_hWnd);

    if (0 == _hDC) {
        destroy();
    }
}

classic_gl_window::~classic_gl_window()
{
    destroy();
}

bool
classic_gl_window::valid() const
{
    return ((0 != _hWnd) && (0 != _hDC));
}

void
classic_gl_window::destroy()
{
    if (0 != _hDC) {
        ::ReleaseDC(_hWnd, _hDC);
        _hDC = NULL;
    }

    if (0 != _hWnd) {
        ::DestroyWindow(_hWnd);
        ::UnregisterClass(_wnd_class_name.c_str(), ::GetModuleHandle(NULL));
        _hWnd = NULL;
    }
}

classic_gl_context::classic_gl_context(const classic_gl_window& in_window)
  : _hGLRC(0),
    _window(in_window)
{
    DWORD dwflags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER_DONTCARE | PFD_STEREO_DONTCARE;

#if SCM_WIN_VER >= SCM_WIN_VER_VISTA
    dwflags |= PFD_SUPPORT_COMPOSITION;
#endif

    PIXELFORMATDESCRIPTOR pfd =  {
        sizeof(PIXELFORMATDESCRIPTOR),
        1,
        dwflags,
        PFD_TYPE_RGBA,
        0,                      // color bits
        0, 0, 0, 0, 0, 0, 0, 0, // color/alpha bits/shift
        0,                      // accum bits
        0, 0, 0, 0,             // accum rgba bits
        0,                      // depth bits
        0,                      // stencil bits
        0,                      // aux buffers
        PFD_MAIN_PLANE,
        0,
        0, 0, 0
    };

    int pfmt = ::ChoosePixelFormat(in_window._hDC, &pfd);

    if (pfmt < 1) {
        destroy();
    }

    if (!::SetPixelFormat(in_window._hDC, pfmt, &pfd)) {
        destroy();
    }

    _hGLRC = ::wglCreateContext(in_window._hDC);

    if (_hGLRC == NULL) {
        destroy();
    }

    ::wglMakeCurrent(in_window._hDC, _hGLRC);
}

classic_gl_context::~classic_gl_context()
{
    if (_hGLRC != NULL) {
        destroy();
    }
}

bool
classic_gl_context::valid() const
{
    return (0 != _hGLRC);
}

void
classic_gl_context::destroy()
{
    if (0 != _hGLRC) {
        ::wglMakeCurrent(_window._hDC, NULL);
        ::wglDeleteContext(_hGLRC);
        _hGLRC = NULL;
    }
}

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
