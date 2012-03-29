
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_HEADLESS_CONTEXT_H_INCLUDED
#define SCM_GL_HEADLESS_CONTEXT_H_INCLUDED

#include <scm/gl_classic/render_context/context_base.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class window_context;

class __scm_export(gl_core) headless_context : public context_base
{
public:
    headless_context();
    virtual ~headless_context();

    virtual bool            setup(const context_format& desc,
                                  const window_context& partent_ctx) = 0;

};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>


#endif // SCM_GL_HEADLESS_CONTEXT_H_INCLUDED
