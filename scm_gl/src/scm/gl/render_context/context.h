
#ifndef SCM_GL__CONTEXT_H_INCLUDED
#define SCM_GL__CONTEXT_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class context : boost::noncopyable
{
public:
    typedef scm::shared_ptr<void>   handle;

public:
    context_base();
    virtual ~context_base();

    virtual void            cleanup() = 0;

    virtual bool            make_current(bool current = true) const = 0;
    bool                    setup_windowed_context(const handle          dev_hndl,
                                                   const handle          wnd_hndl,
                                                   const context_format& desc,
                                                   const context_base&   share_ctx);
    bool                    setup_headless_context(const handle          dev_hndl,
                                                   const context_format& desc,
                                                   const context_base&   parent_ctx);

    bool                    make_current(bool current = true) const;
    void                    swap_buffers(int interval = 0) const;

    void                    cleanup();

    const context_format&   format() const;

    const handle&           context_handle() const;
    const handle&           device_handle() const;
    const handle&           drawable_handle() const;

    virtual bool            empty() const;

protected:
    context_format          _context_format;

    handle                  _device;    // win/WGL: HDC,            linux/GLX: Display*
    handle                  _drawable;  // win/WGL: HWND, HPBUFFER  linux/GLX: Window, GLXPbuffer
    handle                  _context;   // win/WGL: HGLRC,          linux/GLX: GLXContext

}; // class context

} // namepspace gl
} // namepspace scm

#endif // SCM_GL__CONTEXT_H_INCLUDED
