
#include "freetype_types.h"

#include <ft2build.h>
#include FT_FREETYPE_H

namespace scm {
namespace gl {
namespace detail {

ft_library::ft_library() : _lib(0) {}

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

ft_face::ft_face() : _face(0) {}

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


ft_stroker::ft_stroker(const ft_library&  lib,
                       unsigned           border_size) : _stroker(0)
{
    FT_Stroker_Set(_stroker, border_size, FT_STROKER_LINECAP_BUTT, FT_STROKER_LINEJOIN_ROUND, 0);
}

ft_stroker::~ft_stroker()
{
    FT_Stroker_Done(_stroker);
}

} // namespace detail
} // namespace gl
} // namespace scm
