
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "freetype_types.h"

#include <exception>
#include <stdexcept>
#include <sstream>

#include <ft2build.h>
#include FT_FREETYPE_H

namespace scm {
namespace gl {
namespace detail {

ft_library::ft_library() : _lib(0)
{
    if (FT_Init_FreeType(&_lib) != 0) {
        throw(std::runtime_error("ft_library::ft_library() unable to open freetype library"));
    }
}

ft_library::~ft_library()
{
    FT_Done_FreeType(_lib);
}

ft_face::ft_face(const ft_library&  lib,
                 const std::string& file)
  : _face(0)
{
    if (FT_New_Face(lib.get_lib(), file.c_str(), 0, &_face) != 0) {
        throw(std::runtime_error(std::string("ft_face::ft_face() unable to open font - ") + file));
    }
}

void
ft_face::set_size(unsigned point_size,
                  unsigned display_dpi)
{
    if (FT_Set_Char_Size(_face, 0, point_size << 6, 0, display_dpi) != 0) {
        std::ostringstream s;
        s << "font_face::font_face(): unable to set character size (size: " << point_size << ")";
        throw(std::runtime_error(s.str()));
    }
}

void
ft_face::load_glyph(char c, unsigned f)
{
    if(FT_Load_Glyph(_face, FT_Get_Char_Index(_face, c), f)) {
                        //FT_LOAD_DEFAULT)) { //| FT_LOAD_TARGET_NORMAL)) {
                        //FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_LIGHT)) {
        std::ostringstream s;
        s << "ft_face::load_glyph: unable to load character glyph (c: " << c << ")";
        throw(std::runtime_error(s.str()));
    }
}

FT_GlyphSlot
ft_face::get_glyph() const
{
    return (_face->glyph);
}

char
ft_face::get_kerning(char l, char r) const
{
    if (_face->face_flags & FT_FACE_FLAG_KERNING) {
        FT_UInt l_glyph_index = FT_Get_Char_Index(_face, l);
        FT_UInt r_glyph_index = FT_Get_Char_Index(_face, r);

        if ((l_glyph_index == 0) || (r_glyph_index == 0)) {
            return (0);
        }
        FT_Vector   delta;
        FT_Get_Kerning(_face, l_glyph_index, r_glyph_index, FT_KERNING_DEFAULT, &delta);
    
        return (static_cast<char>(delta.x >> 6));
    }
    else {
        return (0);
    }
}

ft_face::~ft_face()
{
    FT_Done_Face(_face);
}


ft_stroker::ft_stroker(const ft_library&  lib,
                       unsigned           border_size) : _stroker(0)
{
    if (FT_Stroker_New(lib.get_lib(), &_stroker) != 0) {
        throw(std::runtime_error("ft_stroker::ft_stroker() unable to create stroker"));
    }
    FT_Stroker_Set(_stroker, border_size, FT_STROKER_LINECAP_BUTT, FT_STROKER_LINEJOIN_ROUND, 0);
}

ft_stroker::~ft_stroker()
{
    FT_Stroker_Done(_stroker);
}

} // namespace detail
} // namespace gl
} // namespace scm
