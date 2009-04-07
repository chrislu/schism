
#include "framebuffer_format.h"

#include <algorithm>

namespace scm {
namespace gl {

framebuffer_format_descriptor::framebuffer_format_descriptor(data_format    color_buffer_format,
                                                             data_format    depth_stencil_buffer_format,
                                                             unsigned       color_buffer_count,
                                                             unsigned       sample_count)
  : _color_buffer_format(color_buffer_format),
    _color_buffer_count(color_buffer_count),
    _depth_stencil_buffer_format(depth_stencil_buffer_format),
    _sample_count(sample_count)
{
}

framebuffer_format_descriptor::framebuffer_format_descriptor(const framebuffer_format_descriptor& fmt)
  : _color_buffer_format(fmt._color_buffer_format),
    _color_buffer_count(fmt._color_buffer_count),
    _depth_stencil_buffer_format(fmt._depth_stencil_buffer_format),
    _sample_count(fmt._sample_count)
{
}

framebuffer_format_descriptor::~framebuffer_format_descriptor()
{
}

framebuffer_format_descriptor&
framebuffer_format_descriptor::operator=(const framebuffer_format_descriptor& rhs)
{
    framebuffer_format_descriptor tmp(rhs);
    swap(tmp);
    return (*this);
}

void
framebuffer_format_descriptor::swap(framebuffer_format_descriptor& fmt)
{
    std::swap(_color_buffer_format,         fmt._color_buffer_format);
    std::swap(_color_buffer_count,          fmt._color_buffer_count);
    std::swap(_depth_stencil_buffer_format, fmt._depth_stencil_buffer_format);
    std::swap(_sample_count,                fmt._sample_count);
}

bool
framebuffer_format_descriptor::operator==(const framebuffer_format_descriptor& fmt) const
{
    bool tmp_ret = true;

    tmp_ret = tmp_ret && (_color_buffer_format          == fmt._color_buffer_format);
    tmp_ret = tmp_ret && (_color_buffer_count           == fmt._color_buffer_count);
    tmp_ret = tmp_ret && (_depth_stencil_buffer_format  == fmt._depth_stencil_buffer_format);
    tmp_ret = tmp_ret && (_sample_count                 == fmt._sample_count);

    return (tmp_ret);
}

bool
framebuffer_format_descriptor::operator!=(const framebuffer_format_descriptor& fmt) const
{
    return (!(*this == fmt));
}

/*static*/
const framebuffer_format_descriptor&
framebuffer_format_descriptor::null_format()
{
    static framebuffer_format_descriptor nullfmt;
    return (nullfmt);
}

/*static*/
const framebuffer_format_descriptor&
framebuffer_format_descriptor::default_format()
{
    static framebuffer_format_descriptor deffmt(FORMAT_RGBA8,
                                                FORMAT_D24_S8,
                                                2,
                                                1);
    return (deffmt);
}

std::ostream& operator<<(std::ostream& out_stream, const framebuffer_format_descriptor& fmt)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "color_buffer_format:         " << fmt._color_buffer_format << std::endl
               << "color_buffer_count:          " << fmt._color_buffer_count << std::endl
               << "depth_stencil_buffer_format: " << fmt._depth_stencil_buffer_format << std::endl
               << "sample_count:                " << fmt._sample_count;

    return (out_stream);
}

} // namespace gl
} // namespace scm
