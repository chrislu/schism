
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED
#define SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED

#include <string>
#include <vector>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_OUTLINE_H
#include FT_STROKER_H
#include FT_LCD_FILTER_H

namespace scm {
namespace gl {
namespace detail {

class ft_library
{
public:
    ft_library();
    /*virtual*/ ~ft_library();

    const FT_Library    get_lib() const { return (_lib); }
protected:
    FT_Library          _lib;
}; // class ft_library

class ft_face
{
public:
    ft_face(const ft_library&  /*lib*/,
            const std::string& /*file*/);
    /*virtual*/ ~ft_face();

    void                set_size(unsigned           /*point_size*/,
                                 unsigned           /*display_dpi*/);
    void                load_glyph(char c, unsigned f);
    FT_GlyphSlot        get_glyph() const;
    char                get_kerning(char l, char r) const;
    const FT_Face       get_face() const { return (_face); }

protected:
    FT_Face             _face;
}; // class ft_face

class ft_stroker
{
public:
    ft_stroker(const ft_library&  /*lib*/,
               unsigned           /*border_size*/);
    /*virtual*/ ~ft_stroker();

    const FT_Stroker    get_stroker() const { return (_stroker); }

protected:
    FT_Stroker      _stroker;
}; // class ft_stroker

} // namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED
