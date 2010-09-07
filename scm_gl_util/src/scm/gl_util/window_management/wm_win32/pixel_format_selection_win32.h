

#ifndef SCM_GL_UTIL_WM_WIN32_PIXEL_FORMAT_SELECTION_WIN32_H_INCLUDED
#define SCM_GL_UTIL_WM_WIN32_PIXEL_FORMAT_SELECTION_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <ostream>

#include <scm/core/pointer_types.h>
#include <scm/gl_util/window_management/wm_win32/wgl_extensions.h>

namespace scm {
namespace gl {
namespace wm {

class display;
struct pixel_format_desc;

namespace util {

class pixel_format_selector
{
public:
    typedef enum {
        window_surface,
        pbuffer_surface
    } surface_type;

    static bool choose(HDC                               in_device,
                       const pixel_format_desc&          in_pfd,
                       const surface_type                in_surface_type,
                       const shared_ptr<wgl_extensions>& in_wgl,
                       int&                              out_pf_num,
                       std::ostream&                     out_stream);

}; // class pixel_format_selector

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WM_WIN32_PIXEL_FORMAT_SELECTION_WIN32_H_INCLUDED
