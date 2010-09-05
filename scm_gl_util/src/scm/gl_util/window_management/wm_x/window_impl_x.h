
#ifndef SCM_GL_UTIL_WM_X_WINDOW_IMPL_H_INCLUDED
#define SCM_GL_UTIL_WM_X_WINDOW_IMPL_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/window.h>

namespace scm {
namespace gl {
namespace wm {

struct window::window_impl
{
    window_impl(const display&           in_display,
                const std::string&       in_title,
                const math::vec2i&       in_position,
                const math::vec2ui&      in_size,
                const pixel_format_desc& in_pf);
    virtual ~window_impl();

    void            show();
    void            hide();

    long            _window_handle;

}; // class window_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_UTIL_WM_X_WINDOW_IMPL_H_INCLUDED
