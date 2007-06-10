
#include "font.h"

#include <boost/functional/hash.hpp>

#include <scm/core.h>
#include <scm/core/exception/system_exception.h>

using namespace scm;
using namespace scm::gl;

// gl_font_descriptor
std::size_t font_descriptor::hash_value() const
{
    std::size_t seed = 0;

    boost::hash_combine<std::string>(seed, _name);
    boost::hash_combine<unsigned>(seed, _size);

    return (seed);
}

// gl_font_face
font_face_resource::~font_face_resource()
{
    cleanup_textures();
}

font_face_resource::font_face_resource(const font_descriptor& desc)
: res::resource<font_descriptor>(desc)
{
}

const texture_2d_rect& font_face_resource::get_glyph_texture(font::face::style_type style) const
{
    style_textur_container::const_iterator style_it = _style_textures.find(style);

    if (style_it == _style_textures.end()) {
        style_it = _style_textures.find(font::face::regular);

        if (style_it == _style_textures.end()) {
            std::stringstream output;

            output << "gl_font_face::get_glyph(): "
                   << "unable to retrieve requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << std::endl;

            console.get() << con::log_level(con::error)
                          << output.str();

            throw scm::core::system_exception(output.str());
        }
    }

    return (*style_it->second.get());
}

void font_face_resource::cleanup_textures()
{
    //style_textur_container::iterator style_it;

    //for (style_it  = _style_textures.begin();
    //     style_it != _style_textures.end();
    //     ++style_it) {
    //    style_it->second.reset();
    //}
    _style_textures.clear();
}
