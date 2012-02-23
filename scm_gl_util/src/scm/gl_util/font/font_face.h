
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_FONT_FACE_H_INCLUDED
#define SCM_GL_UTIL_FONT_FACE_H_INCLUDED

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <boost/multi_array.hpp>

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/font/font_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) font_face
{
public:
    typedef enum {
        style_regular       = 0x00,
        style_italic,
        style_bold,
        style_bold_italic,

        style_count
    } style_type;
    typedef enum {
        smooth_normal   = 0x00,
        smooth_lcd,

        smooth_count
    } smooth_type;
    struct glyph_info {
        math::vec2f    _texture_origin;
        math::vec2f    _texture_box_size;

        math::vec2i    _box_size;
        math::vec2i    _border_bearing;
        unsigned       _advance;
        math::vec2i    _bearing;

        glyph_info()
          : _texture_origin(math::vec2f::zero())
          , _texture_box_size(math::vec2f::zero())
          , _box_size(math::vec2i::zero())
          , _border_bearing(math::vec2i::zero())
          , _advance(0)
          , _bearing(math::vec2i::zero())
        {
        }
    }; // struct glyph_info

    static const unsigned       min_char = 32u;
    static const unsigned       max_char = 128u;

    static const unsigned       default_point_size   = 12;
    //static const float          default_border_size  = 0.0f;
    static const unsigned       default_display_dpi  = 72;
    static const smooth_type    default_smooth_style = smooth_normal;

protected:
    typedef std::vector<glyph_info>     glyph_container;
    typedef boost::multi_array<char, 2> kerning_table;
    struct font_style {
        glyph_container _glyphs;
        kerning_table   _kerning_table;
        int             _underline_position;
        unsigned        _underline_thickness;
        unsigned        _line_spacing;
    }; // struct style_info
    typedef std::vector<font_style>     style_container;

public:
    font_face(const render_device_ptr& device,                  
              const std::string&       font_file,
              unsigned                 point_size  = default_point_size,
              float                    border_size = 0.0f,//default_border_size,
              smooth_type              smooth_type = default_smooth_style,
              unsigned                 display_dpi = default_display_dpi);
    virtual ~font_face();

    const std::string&              name() const;
    unsigned                        point_size() const;
    unsigned                        border_size() const;
    unsigned                        dpi() const;
    smooth_type                     smooth_style() const;
    bool                            has_style(style_type s) const;

    const glyph_info&               glyph(char c, style_type s = style_regular) const;
    unsigned                        line_advance(style_type s = style_regular) const;
    int                             kerning(char l, char r, style_type s = style_regular) const;

    int                             underline_position(style_type s = style_regular) const;
    int                             underline_thickness(style_type s = style_regular) const;

    const texture_2d_ptr&           styles_texture_array() const;
    const texture_2d_ptr&           styles_border_texture_array() const;

protected:
    void                            render_glyphs();

    void                            cleanup();

protected:
    style_container                 _font_styles;
    std::vector<bool>               _font_styles_available;
    texture_2d_ptr                  _font_styles_texture_array;
    texture_2d_ptr                  _font_styles_border_texture_array;
    smooth_type                     _font_smooth_style;

    std::string                     _name;
    unsigned                        _point_size;
    unsigned                        _border_size;
    unsigned                        _dpi;

}; // class font_face

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_FONT_FACE_H_INCLUDED
