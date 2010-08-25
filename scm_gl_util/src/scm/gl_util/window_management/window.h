
#ifndef SCM_GL_UTIL_WM_WINDOW_H_INCLUDED
#define SCM_GL_UTIL_WM_WINDOW_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class window
{
public:
    window();
    virtual ~window();

private:
    struct window_impl;
    shared_ptr<window_impl>     _impl;

private:
    // non_copyable
    window(const window&);
    window& operator=(const window&);

}; // class window

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_WINDOW_H_INCLUDED
