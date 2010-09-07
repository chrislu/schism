
#ifndef SCM_GL_UTIL_WM_CONTEXT_H_INCLUDED
#define SCM_GL_UTIL_WM_CONTEXT_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/pixel_format.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class display;
class surface;

class __scm_export(gl_util) context
{
public:
    context(const surface&  in_surface,
            const context&  in_share_ctx);
    virtual ~context();

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

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_CONTEXT_H_INCLUDED
