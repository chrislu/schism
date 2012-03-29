
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL__CONTEXT_H_INCLUDED
#define SCM_GL__CONTEXT_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl_classic {



class context : boost::noncopyable
{
public:
    typedef scm::shared_ptr<void>   handle;

#if   SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    typedef handle          device_handle;
    typedef handle          surface_handle;
#elif SCM_PLATFORM == SCM_PLATFORM_LINUX
    typedef Display*        device_handle;
    typedef XID             surface_handle;
#elif SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif // SCM_PLATFORM

    struct surface_descriptor
    {
        device_handle           _device;
        surface_handle          _surface;
    }; // struct surface_descriptor

public:
    context(const surface_descriptor&  srfce,
            const context_format&      desc,
            const context_base&        share_ctx = null_context());
    virtual ~context();

    virtual bool            make_current(bool current = true) const = 0;
    virtual void            swap_buffers(int interval = 0) const = 0;
    virtual bool            empty() const = 0;

    const context_format&   format() const;
    static context&         null_context();

protected:
    context(); // create null context

    context_format          _context_format;

    handle                  _device;    // win/WGL: HDC,            linux/GLX: Display*
    handle                  _drawable;  // win/WGL: HWND, HPBUFFER  linux/GLX: Window, GLXPbuffer
    handle                  _context;   // win/WGL: HGLRC,          linux/GLX: GLXContext

}; // class context

} // namepspace gl
} // namepspace scm

#endif // SCM_GL__CONTEXT_H_INCLUDED
