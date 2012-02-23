
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WINDOW_H_INCLUDED
#define SCM_GL_CORE_WM_WINDOW_H_INCLUDED

#include <string>

#include <scm/core/math.h>
#include <scm/core/memory.h>

#include <scm/gl_core/window_management/wm_fwd.h>
#include <scm/gl_core/window_management/surface.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
struct HWND__;
typedef struct HWND__* HWND;
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

namespace scm {
namespace gl {
namespace wm {

class __scm_export(gl_core) window : public surface
{
public:
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    typedef HWND handle;
#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
    typedef unsigned long handle;
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif

public:
    window(const display_cptr&      in_display,
           const std::string&       in_title,
           const math::vec2i&       in_position,
           const math::vec2ui&      in_size,
           const format_desc&       in_sf);
    window(const display_cptr&      in_display,
           const handle             in_parent,
           const std::string&       in_title,
           const math::vec2i&       in_position,
           const math::vec2ui&      in_size,
           const format_desc&       in_sf);
    virtual ~window();

    void                        swap_buffers(int interval = 0) const;
    const handle                window_handle() const;

    void                        show();
    void                        hide();

protected:
    struct window_impl;

private:
    // non_copyable
    window(const window&);
    window& operator=(const window&);
}; // class window

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_WM_WINDOW_H_INCLUDED
