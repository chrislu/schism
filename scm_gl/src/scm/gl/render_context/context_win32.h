
#ifndef SCM_GL_CONTEXT_WIN32_H_INCLUDED
#define SCM_GL_CONTEXT_WIN32_H_INCLUDED

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/gl/render_context/context.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) context_win32 : public context
{
public:
    context_win32();
    virtual ~context_win32();

    bool            setup(const wnd_handle hwnd,
                          const context_format& desc);
    bool            setup(const wnd_handle hwnd,
                          const context_format& desc,
                          const context& share_ctx);
    void            cleanup();

    bool            make_current(bool current = true) const;
    void            swap_buffers() const;

    static context_win32& empty_context();

protected:
    handle          _hDC;
    handle          _wnd_handle;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CONTEXT_WIN32_H_INCLUDED
