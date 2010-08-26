
#ifndef SCM_GL_UTIL_WM_WINDOW_H_INCLUDED
#define SCM_GL_UTIL_WM_WINDOW_H_INCLUDED

#include <string>

#include <scm/core/math.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/pixel_format.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
struct HWND__;
typedef struct HWND__* HWND;
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

namespace scm {
namespace gl {
namespace wm {

class display;

class window
{
public:
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    typedef HWND wnd_handle;
#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif


public:
    window(const display&           in_display,
           const std::string&       in_title,
           const math::vec2ui&      in_size,
           const pixel_format_desc& in_pf);
    virtual ~window();

    const wnd_handle            window_handle() const;
    const display&              associated_display() const;
    const pixel_format_desc&    pixel_format() const;
   
    void                        show();
    void                        hide();

private:
    struct window_impl;
    shared_ptr<window_impl>     _impl;

    const display&              _associated_display;
    pixel_format_desc           _pixel_format;

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
