
#ifndef SCM_GL_UTIL_WINDOW_CONTEXT_WIN32_H_INCLUDED
#define SCM_GL_UTIL_WINDOW_CONTEXT_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/pointer_types.h>

#include <scm/gl_util/render_context/detail/wgl.h>
#include <scm/gl_util/render_context/window_context.h>

#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) window_context_win32 : public window_context
{
public:
    window_context_win32();
    virtual ~window_context_win32();

    bool            setup(const wnd_handle hwnd,
                          const context_format& desc);
    bool            setup(const wnd_handle hwnd,
                          const context_format& desc,
                          const context_base& share_ctx);

    void            cleanup();

    bool            make_current(bool current = true) const;
    void            swap_buffers(int interval = 0) const;

    static window_context_win32& empty_context();

    const shared_ptr<detail::wgl> wgl() const;

protected:
    handle                  _wnd_handle;
    shared_ptr<detail::wgl> _wgl;

    bool                    _swap_control_supported;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WINDOW_CONTEXT_WIN32_H_INCLUDED
