
#ifndef SCM_GL_UTIL_CONTEXT_H_INCLUDED
#define SCM_GL_UTIL_CONTEXT_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class display;
class pixel_format_desc;
class surface;

class context
{
public:

public:
    context(const display&            in_display,
            const pixel_format_desc&  in_pixel_format,
            const context&            in_share_ctx);
    virtual ~context();

    bool                        make_current(const surface& cur_surface) const;
    bool                        make_current(const surface& draw_surface,
                                             const surface& read_surface) const;

    const display&              associated_display() const;
    const pixel_format_desc&    pixel_format() const;

private:
    struct context_impl;
    shared_ptr<context_impl>    _impl;

    const display&              _associated_display;
    pixel_format_desc           _pixel_format;

private:
    // non_copyable
    context(const context&);
    context& operator=(const context&);

}; // class context

} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_CONTEXT_H_INCLUDED
