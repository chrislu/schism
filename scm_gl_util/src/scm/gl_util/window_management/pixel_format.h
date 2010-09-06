
#ifndef SCM_GL_UTIL_WM_PIXEL_FORMAT_H_INCLUDED
#define SCM_GL_UTIL_WM_PIXEL_FORMAT_H_INCLUDED

#include <ostream>

#include <scm/gl_core/data_formats.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

struct __scm_export(gl_util) pixel_format_desc
{
    data_format             _color_format;
    data_format             _depth_stencil_format;

    bool                    _double_buffer;
    bool                    _quad_buffer_stereo;

    pixel_format_desc(data_format color_fmt, data_format depth_stencil_fmt,
                      bool double_buffer, bool quad_buffer_stereo = false);
    /*virtual*/ ~pixel_format_desc();

    friend __scm_export(gl_util) std::ostream& operator<<(std::ostream& out_stream, const pixel_format_desc& pf);

}; // class pixel_format_desc

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_PIXEL_FORMAT_H_INCLUDED
