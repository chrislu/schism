
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_TEXT_H_INCLUDED
#define SCM_GL_UTIL_TEXT_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) text
{
public:
    text(const render_device_ptr&    device,
         const font_face_cptr&       font,
         const font_face::style_type stl,
         const std::string&          str);
    virtual ~text();

    const font_face_cptr&       font() const;
    const font_face::style_type text_style() const;
    const std::string&          text_string() const;
    void                        text_string(const std::string& str);
    void                        text_string(const std::string& str,
                                            const font_face::style_type stl);
    bool                        text_kerning() const;
    void                        text_kerning(bool k);

    const math::vec4f&          text_color() const;
    void                        text_color(const math::vec4f& c) ;
    const math::vec4f&          text_outline_color() const;
    void                        text_outline_color(const math::vec4f& c) ;
    const math::vec4f&          text_shadow_color() const;
    void                        text_shadow_color(const math::vec4f& c);
    const math::vec2i&          text_shadow_offset() const;
    void                        text_shadow_offset(const math::vec2i& o);

    const math::vec2i&          text_bounding_box() const;

protected:
    void                        update();

protected:
    font_face_cptr              _font;
    font_face::style_type       _text_style;
    std::string                 _text_string;
    bool                        _text_kerning;

    math::vec4f                 _text_color;
    math::vec4f                 _text_outline_color;
    math::vec4f                 _text_shadow_color;
    math::vec2i                 _text_shadow_offset;

    math::vec2i                 _text_bounding_box;

    int                         _glyph_capacity;
    buffer_ptr                  _vertex_buffer;
    buffer_ptr                  _index_buffer;
    int                         _indices_count;
    primitive_topology          _topology;

    vertex_array_ptr            _vertex_array;

    render_device_wptr          _render_device;
    render_context_wptr         _render_context;

    friend class text_renderer;
}; // class text

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXT_H_INCLUDED
