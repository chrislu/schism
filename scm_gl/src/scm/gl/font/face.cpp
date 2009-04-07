
#include "face.h"

#include <stdexcept>

#include <scm/core.h>

namespace scm {
namespace gl {

// gl_font_face
face::face()
{
}

face::~face()
{
    cleanup_textures();
}

const texture_2d_rect& face::get_glyph_texture(font::face::style_type style) const
{
    style_textur_container::const_iterator style_it = _style_textures.find(style);

    if (style_it == _style_textures.end()) {
        style_it = _style_textures.find(font::face::regular);

        if (style_it == _style_textures.end()) {
            std::stringstream output;

            output << "scm::gl::face::get_glyph(): "
                   << "unable to retrieve requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << std::endl;

            scm::err() << scm::log_level(scm::logging::ll_error)
                       << output.str();

            throw std::runtime_error(output.str());
        }
    }

    return (*style_it->second.get());
}

void face::cleanup_textures()
{
    //style_textur_container::iterator style_it;

    //for (style_it  = _style_textures.begin();
    //     style_it != _style_textures.end();
    //     ++style_it) {
    //    style_it->second.reset();
    //}
    _style_textures.clear();
}

} // namespace gl
} // namespace scm
