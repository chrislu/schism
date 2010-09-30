
#ifndef SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED
#define SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED

#include <string>

#include <ft2build.h>
#include FT_FREETYPE_H

namespace scm {
namespace gl {
namespace detail {

class ft_library
{
public:
    ft_library();
    virtual ~ft_library();

    bool                open();
    const FT_Library    get_lib() const { return (_lib); }

protected:
    FT_Library          _lib;
}; // class ft_library

class ft_face
{
public:
    ft_face();
    virtual ~ft_face();

    bool                open_face(const ft_library&  /*lib*/,
                                  const std::string& /*file*/);

    const FT_Face       get_face() const { return (_face); }

protected:
    FT_Face             _face;
}; // class ft_face

} // namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED
