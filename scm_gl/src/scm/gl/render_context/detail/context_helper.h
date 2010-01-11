
#ifndef SCM_GL_DETAIL_GL_CONTEXT_HELPER_CONTEXT_HELPER_H_INCLUDED
#define SCM_GL_DETAIL_GL_CONTEXT_HELPER_CONTEXT_HELPER_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <string>

#include <scm/core/platform/windows.h>

namespace scm {
namespace gl {

namespace detail {

class classic_gl_context
{
public:
    classic_gl_context();
    virtual ~classic_gl_context();
    bool create();
    void destroy();

private:
    HWND    _hWnd;
    HDC     _hDC;
    HGLRC   _hGLRC;

    std::string _wnd_class_name;
};

} // namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#endif // SCM_GL_DETAIL_GL_CONTEXT_HELPER_CONTEXT_HELPER_H_INCLUDED
