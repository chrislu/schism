
#ifndef SCM_GL_UTIL_WM_DISPLAY_H_INCLUDED
#define SCM_GL_UTIL_WM_DISPLAY_H_INCLUDED

#include <scm/core/memory.h>

#include <scm/gl_util/window_management/wm_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class __scm_export(gl_util) display
{
public:
    display(const std::string& name);
    virtual ~display();

private:
    struct display_impl;
    shared_ptr<display_impl>    _impl;

private:
    // non_copyable
    display(const display&);
    display& operator=(const display&);

    friend class scm::gl::wm::context;
    friend class scm::gl::wm::window;
    friend class scm::gl::wm::headless_surface;
}; // class display

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_DISPLAY_H_INCLUDED
