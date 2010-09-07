
#ifndef SCM_GL_UTIL_WM_SURFACE_H_INCLUDED
#define SCM_GL_UTIL_WM_SURFACE_H_INCLUDED

#include <scm/gl_util/window_management/pixel_format.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class display;

class __scm_export(gl_util) surface
{
public:
    virtual ~surface();

    //virtual bool                make_current(bool current = true) const = 0;
    //virtual void                swap_buffers(int interval = 0) const = 0;

    const display&              associated_display() const;
    const pixel_format_desc&    pixel_format() const;

private:
    const display&              _associated_display;
    pixel_format_desc           _pixel_format;

protected:
    surface(const display&           in_display,
            const pixel_format_desc& in_pf);
private:
    // non_copyable
    surface(const surface&);
    surface& operator=(const surface&);
}; // class surface

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_SURFACE_H_INCLUDED
