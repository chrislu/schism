
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_WINDOW_CONTEXT_WIN32_H_INCLUDED
#define SCM_GL_WINDOW_CONTEXT_WIN32_H_INCLUDED

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/gl_classic/render_context/window_context.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) window_context_win32 : public window_context
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

protected:
    handle          _wnd_handle;

    bool            _swap_control_supported;
};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_WINDOW_CONTEXT_WIN32_H_INCLUDED
