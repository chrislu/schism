
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_HEADLESS_CONTEXT_WIN32_H_INCLUDED
#define SCM_GL_HEADLESS_CONTEXT_WIN32_H_INCLUDED

#include <scm/gl_classic/render_context/headless_context.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) headless_context_win32 : public headless_context
{
public:
    headless_context_win32();
    virtual ~headless_context_win32();

    bool            setup(const context_format& desc,
                          const window_context& partent_ctx);
    void            cleanup();

    bool            make_current(bool current = true) const;

protected:
    handle          _pbuffer;

};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_HEADLESS_CONTEXT_WIN32_H_INCLUDED
