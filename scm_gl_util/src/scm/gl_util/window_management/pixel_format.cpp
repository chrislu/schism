
#include "pixel_format.h"

namespace scm {
namespace gl {
namespace wm {

pixel_format_desc::pixel_format_desc(data_format color_fmt, data_format depth_stencil_fmt,
                                     bool double_buffer, bool quad_buffer_stereo)
  : _color_format(color_fmt),
    _depth_stencil_format(depth_stencil_fmt),
    _double_buffer(double_buffer),
    _quad_buffer_stereo(quad_buffer_stereo)
{
}

pixel_format_desc::~pixel_format_desc()
{
}

std::ostream& operator<<(std::ostream& out_stream, const pixel_format_desc& fmt)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "color_buffer_format: " << format_string(fmt._color_format)         << std::endl
               << "color_buffer_count:  " << format_string(fmt._depth_stencil_format) << std::endl
               << "double_buffer:       " << fmt._double_buffer                       << std::endl
               << "quad_buffer_stereo:  " << fmt._quad_buffer_stereo;

    return (out_stream);
}

} // namespace wm
} // namepspace gl
} // namepspace scm
