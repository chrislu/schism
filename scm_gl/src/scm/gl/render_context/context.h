
#ifndef SCM_GL_CONTEXT_H_INCLUDED
#define SCM_GL_CONTEXT_H_INCLUDED

#include <boost/utility.hpp>

#include <scm/core/pointer_types.h>
#include <scm/gl/render_context/context_format.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class context_format;

class __scm_export(ogl) context : boost::noncopyable
{
public:
    typedef void*                   wnd_handle;
    typedef scm::shared_ptr<void>   handle;

public:
    context();
    virtual ~context();

    virtual bool            setup(const wnd_handle hwnd,
                                  const context_format& desc) = 0;
    virtual bool            setup(const wnd_handle hwnd,
                                  const context_format& desc,
                                  const context& share_ctx) = 0;
    virtual void            cleanup() = 0;

    virtual bool            make_current(bool current = true) const = 0;
    virtual void            swap_buffers() const = 0;

    const context_format&   format() const;
    const handle&           context_handle() const;

    virtual bool            operator==(const context& rhs) const;
    virtual bool            operator!=(const context& rhs) const;

    virtual bool            empty() const;

protected:
    context_format          _context_format;
    handle                  _context_handle;

};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CONTEXT_H_INCLUDED
