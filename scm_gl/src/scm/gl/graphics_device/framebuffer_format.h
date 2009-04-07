
#ifndef SCM_GL_FRAMEBUFFER_FORMAT_FRAMEBUFFER_FORMAT_H_INCLUDED
#define SCM_GL_FRAMEBUFFER_FORMAT_FRAMEBUFFER_FORMAT_H_INCLUDED

#include <ostream>

#include <scm/core/math.h>
#include <scm/gl/graphics_device/formats.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(ogl) framebuffer_format_descriptor
{
    data_format     _color_buffer_format;           // color buffer mode
    data_format     _depth_stencil_buffer_format;   // depth, stencil buffer format
    unsigned        _color_buffer_count;            // number of color buffers in swap chain
    unsigned        _sample_count;                  // multi sample mode (1x default,...)

    framebuffer_format_descriptor(data_format   color_buffer_format          = FORMAT_NULL,
                                  data_format   depth_stencil_buffer_format  = FORMAT_NULL,
                                  unsigned      color_buffer_count           = 1,
                                  unsigned      sample_count                 = 1);
    framebuffer_format_descriptor(const framebuffer_format_descriptor& fmt);
    /*virtual*/ ~framebuffer_format_descriptor();

    framebuffer_format_descriptor&  operator=(const framebuffer_format_descriptor& rhs);
    void                            swap(framebuffer_format_descriptor& fmt);

    bool                            operator==(const framebuffer_format_descriptor& fmt) const;
    bool                            operator!=(const framebuffer_format_descriptor& fmt) const;

    static const framebuffer_format_descriptor& null_format();
    static const framebuffer_format_descriptor& default_format();

    friend __scm_export(ogl) std::ostream& operator<<(std::ostream& out_stream, const framebuffer_format_descriptor& fmt);

}; // struct framebuffer_format_descriptor

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_FRAMEBUFFER_FORMAT_FRAMEBUFFER_FORMAT_H_INCLUDED
