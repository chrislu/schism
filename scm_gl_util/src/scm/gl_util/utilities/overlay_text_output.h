
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_OVERLAY_TEXT_OUTPUT_H_INCLUDED
#define SCM_GL_UTIL_OVERLAY_TEXT_OUTPUT_H_INCLUDED

#include <iosfwd>
#include <cassert>
#include <string>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace util {

class __scm_export(gl_util) overlay_text_output
{
public:
    overlay_text_output(const gl::render_device_ptr& device,
                        const math::vec2ui&          vp_size,
                        const int                    text_size);
    virtual ~overlay_text_output();

    const math::vec4f&                  back_color() const;
    void                                back_color(const math::vec4f& c);
    const math::vec4f&                  frame_color() const;
    void                                frame_color(const math::vec4f& c);

    const math::vec4f&                  text_color() const;
    void                                text_color(const math::vec4f& c);
    const math::vec4f&                  text_outline_color() const;
    void                                text_outline_color(const math::vec4f& c);

    void                                update(const gl::render_context_ptr& context,
                                               const std::string&            text,
                                               const math::vec2i&            out_pos);
    void                                update(const gl::render_context_ptr& context,
                                               const std::string&            text);
    void                                draw(const gl::render_context_ptr& context);
protected:
    math::vec2ui                        _viewport_size;

    math::vec4f                         _back_color;
    math::vec4f                         _frame_color;

    gl::text_renderer_ptr               _text_renderer;
    gl::text_ptr                        _output_text;
    int                                 _output_text_size;
    math::vec2i                         _output_text_pos;
    math::vec2i                         _output_text_frame_size;
    gl::quad_geometry_ptr               _output_text_background;
    gl::geometry_highlight_ptr          _geom_highlighter;

}; // class overlay_text_output

} // namespace util
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //SCM_GL_UTIL_OVERLAY_TEXT_OUTPUT_H_INCLUDED
