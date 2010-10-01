
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
    struct glyph_info {
        math::vec2f    _tex_lower_left;
        math::vec2f    _tex_upper_right;

        unsigned       _advance;
        math::vec2i    _bearing;
    }; // struct glyph_info

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
              unsigned                 point_size  = 12,
              unsigned                 display_dpi = 72);
    virtual ~font_face();

    const std::string&              name() const;
    unsigned                        size_at_72dpi() const;
    bool                            has_style(style_type s) const;

    const glyph_info&               glyph(char c, style_type s = style_regular) const;
    unsigned                        line_spacing(style_type s = style_regular) const;
    int                             kerning(char l, char r, style_type s = style_regular) const;

    int                             underline_position(style_type s = style_regular) const;
    int                             underline_thickness(style_type s = style_regular) const;

    const texture_2d_ptr&           styles_texture_array() const;

protected:
    void                            cleanup();

protected:
    style_container                 _font_styles;
    std::vector<bool>               _font_styles_available;
    texture_2d_ptr                  _font_styles_texture_array;

    std::string                     _name;
    unsigned                        _size_at_72dpi;

}; // class font_face

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_FONT_FACE_H_INCLUDED
