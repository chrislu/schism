
#include "freetype_types.h"

#include <sstream>

#include <scm/console.h>
#include <scm/core/exception/system_exception.h>

#include <ft2build.h>
#include FT_FREETYPE_H

using namespace scm::font::detail;

ft_library::ft_library(){}

bool ft_library::open()
{
    if (FT_Init_FreeType(&_lib) != 0) {
        return (false);
    }
    
    return (true);
}

ft_library::~ft_library()
{
    FT_Done_FreeType(_lib);
}

ft_face::ft_face(){}

bool ft_face::open_face(const ft_library&  lib,
                        const std::string& file)
{
    if (FT_New_Face(lib.get_lib(), file.c_str(), 0, &_face) != 0) {
        return (false);
    }
    
    return (true);
}

ft_face::~ft_face()
{
    FT_Done_Face(_face);
}
