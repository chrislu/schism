
#ifndef SCM_GL_UTIL_WM_PIXEL_FORMAT_H_INCLUDED
#define SCM_GL_UTIL_WM_PIXEL_FORMAT_H_INCLUDED

#include <scm/gl_core/data_formats.h>

namespace scm {
namespace gl {
namespace wm {

struct pixel_format
{
    data_format             _color_format;
    data_format             _depth_stencil_foramt;

    bool                    _double_buffer;
    bool                    _quad_buffer_stereo;

}; // class pixel_format

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_PIXEL_FORMAT_H_INCLUDED
