
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CONTEXT_CORE_H_INCLUDED
#define SCM_GL_CONTEXT_CORE_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl_classic {

class context_core : boost::noncopyable
{
public:
    typedef scm::shared_ptr<void>   handle;

#if   SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#elif SCM_PLATFORM == SCM_PLATFORM_LINUX
#elif SCM_PLATFORM == SCM_PLATFORM_APPLE
#endif // SCM_PLATFORM

public:
    context_core();
    virtual ~context_core();

    virtual void            cleanup() = 0;

    virtual bool            setup_windowed_context(const handle          dev_hndl,
                                                   const handle          wnd_hndl,
                                                   const context_format& desc,
                                                   const context_core&   share_ctx) = 0;
    virtual bool            setup_headless_context(const context_core&   parent_ctx) = 0;
    void                    cleanup();

    virtual bool            make_current(bool current = true) const;
    virtual void            swap_buffers(int interval = 0) const;


    const handle&           device_handle() const;
    const handle&           drawable_handle() const;
    const handle&           context_handle() const;

    virtual bool            empty() const;

protected:
    context_format          _context_format;

    handle                  _device;    // win/WGL: HDC,            linux/GLX: Display*
    handle                  _drawable;  // win/WGL: HWND, HPBUFFER  linux/GLX: Window, GLXPbuffer
    handle                  _context;   // win/WGL: HGLRC,          linux/GLX: GLXContext

}; // class context_core

} // namepspace gl
} // namepspace scm


#endif // SCM_GL_CONTEXT_CORE_H_INCLUDED
