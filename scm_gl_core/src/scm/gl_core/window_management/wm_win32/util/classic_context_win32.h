
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WIN32_CLASSIC_CONTEXT_WIN32_H_INCLUDED
#define SCM_GL_CORE_WM_WIN32_CLASSIC_CONTEXT_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <string>

#include <scm/core/math.h>
#include <scm/core/platform/windows.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

struct classic_gl_window
{
    classic_gl_window(const math::vec2i&       in_position,
                      const math::vec2ui&      in_size);
    ~classic_gl_window();

    bool    valid() const;
    void    destroy();

    HWND    _hWnd;
    HDC     _hDC;

    std::string _wnd_class_name;
}; // class classic_gl_window

struct classic_gl_context
{
    classic_gl_context(const classic_gl_window& in_window);
    virtual ~classic_gl_context();

    bool    valid() const;
    void    destroy();

    HGLRC                       _hGLRC;
    const classic_gl_window&    _window;
}; // classic_gl_context

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CORE_WM_WIN32_CLASSIC_CONTEXT_WIN32_H_INCLUDED
