
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "surface.h"

#include <cassert>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/context.h>

namespace scm {
namespace gl {
namespace wm {

surface::format_desc::format_desc(data_format color_fmt, data_format depth_stencil_fmt,
                                  bool double_buffer, bool quad_buffer_stereo)
  : _color_format(color_fmt),
    _depth_stencil_format(depth_stencil_fmt),
    _double_buffer(double_buffer),
    _quad_buffer_stereo(quad_buffer_stereo)
{
}

std::ostream& operator<<(std::ostream& out_stream, const surface::format_desc& fmt)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "color_buffer_format: " << format_string(fmt._color_format)         << std::endl
               << "color_buffer_count:  " << format_string(fmt._depth_stencil_format) << std::endl
               << "double_buffer:       " << fmt._double_buffer                       << std::endl
               << "quad_buffer_stereo:  " << fmt._quad_buffer_stereo;

    return out_stream;
}


surface::surface(const display_cptr& in_display,
                 const format_desc&  in_sf)
  : _associated_display(in_display),
    _format(in_sf)
{
}

surface::~surface()
{
}

const display_cptr&
surface::associated_display() const
{
    return _associated_display;
}

const surface::format_desc&
surface::surface_format() const
{
    return _format;
}

/*static*/
const surface::format_desc&
surface::default_format()
{
    static format_desc  default_fmt(FORMAT_RGBA_8, FORMAT_D24_S8, true, false);
    return default_fmt;
}

} // namespace wm
} // namepspace gl
} // namepspace scm
