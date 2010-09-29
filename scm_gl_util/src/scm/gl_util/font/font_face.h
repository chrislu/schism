
#ifndef SCM_GL_UTIL_FONT_FACE_H_INCLUDED
#define SCM_GL_UTIL_FONT_FACE_H_INCLUDED

#include <cstddef>
#include <map>
#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
#if 0
class __scm_export(gl_util) font_face
{
public:
    struct glyph_info {
        math::vec2i    _tex_lower_left;
        math::vec2i    _tex_upper_right;

        unsigned       _advance;
        math::vec2i    _bearing;
    }; // struct glyph
    typedef std::vector<glyph_info>     glyph_container;
    typedef 
    struct style_info {
        character_glyph_mapping         _glyph_mapping;
        kerning_table                   _kerning_table;
        int                             _underline_position;
        unsigned                        _underline_thickness;
        unsigned                        _line_spacing;
    }; // struct style_info

    typedef std::map<unsigned, glyph_info>          character_glyph_mapping;
    //typedef boost::multi_array<char, 2>             kerning_table;

public:
    face_style();
    virtual ~face_style();

    const glyph&                    get_glyph(unsigned char /*ind*/) const;
    unsigned                        get_line_spacing() const;
    int                             get_kerning(unsigned char /*left*/,
                                                unsigned char /*right*/) const;

    int                             underline_position() const;
    int                             underline_thickness() const;

    void                            clear();

protected:

private:
    friend class face_loader;

}; // class face_style
protected:
    typedef boost::shared_ptr<texture_2d_rect>      texture_ptr;
    typedef std::map<font::face::style_type,
                     texture_ptr>                   style_textur_container;

public:
    face();
    virtual ~face();

    const texture_2d_rect&          get_glyph_texture(font::face::style_type /*style*/ = regular) const;

protected:
    void                            cleanup_textures();

    style_textur_container          _style_textures;

private:
    friend class face_loader;

}; // class font_face
#endif
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_FONT_FACE_H_INCLUDED
