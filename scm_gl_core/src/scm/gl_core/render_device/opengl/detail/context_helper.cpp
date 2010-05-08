
#include "context_helper.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <GL/GL.h>

namespace scm {
namespace gl {
namespace detail {

classic_gl_context::classic_gl_context()
  : _hWnd(NULL),
    _hDC(NULL),
    _hGLRC(NULL),
    _wnd_class_name("gl_dummy_for_wgl_stuff")
{
}

classic_gl_context::~classic_gl_context()
{
    if (_hGLRC != NULL) {
        destroy();
    }
}

bool
classic_gl_context::create()
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

void
classic_gl_context::destroy()
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

} // namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
