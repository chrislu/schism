
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CONTEXT_BASE_H_INCLUDED
#define SCM_GL_CONTEXT_BASE_H_INCLUDED

#include <boost/utility.hpp>

#include <scm/core/pointer_types.h>
#include <scm/gl_classic/render_context/context_format.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class context_format;

class __scm_export(gl_core) context_base : boost::noncopyable
{
public:
    typedef scm::shared_ptr<void>   handle;

public:
    context_base();
    virtual ~context_base();

    virtual void            cleanup() = 0;

    virtual bool            make_current(bool current = true) const = 0;

    const context_format&   format() const;
    const handle&           context_handle() const;
    const handle&           device_handle() const;

    virtual bool            empty() const;

protected:
    context_format          _context_format;

    handle                  _device_handle;  // win/WGL: HDC, linux/GLX: Display*
    handle                  _context_handle;

};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>


#endif // SCM_GL_CONTEXT_BASE_H_INCLUDED
