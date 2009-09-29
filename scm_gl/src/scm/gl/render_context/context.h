
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
    typedef void*                   window_handle;

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

} // namepspace gl
} // namepspace scm

#endif // SCM_GL__CONTEXT_H_INCLUDED
