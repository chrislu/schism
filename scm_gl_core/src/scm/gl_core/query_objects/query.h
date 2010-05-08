
#ifndef SCM_GL_CORE_QUERY_H_INCLUDED
#define SCM_GL_CORE_QUERY_H_INCLUDED

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) query : public render_device_child
{
public:
    virtual ~query();

protected:
    query(render_device& in_device);

    void            begin(const render_context& in_context) const;
    void            end(const render_context& in_context) const;

    virtual void    collect(const render_context& in_context) = 0;

    unsigned        query_id() const;
    unsigned        query_type() const;
    
protected:
    unsigned        _gl_query_id;
    unsigned        _gl_query_type;

private:
    friend class render_device;
    friend class render_context;
}; // class query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_QUERY_H_INCLUDED
