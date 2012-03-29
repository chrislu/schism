
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_WINDOW_CONTEXT_H_INCLUDED
#define SCM_GL_WINDOW_CONTEXT_H_INCLUDED

#include <scm/gl_classic/render_context/context_base.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class context_format;

class __scm_export(gl_core) window_context : public context_base
{
public:
    typedef void*                   wnd_handle;

public:
    window_context();
    virtual ~window_context();

    virtual bool            setup(const wnd_handle hwnd,
                                  const context_format& desc) = 0;
    virtual bool            setup(const wnd_handle hwnd,
                                  const context_format& desc,
                                  const context_base& share_ctx) = 0;

    virtual void            swap_buffers(int interval = 0) const = 0;

    virtual bool            operator==(const window_context& rhs) const;
    virtual bool            operator!=(const window_context& rhs) const;

};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_WINDOW_CONTEXT_H_INCLUDED
