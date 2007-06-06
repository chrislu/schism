
#include "face.h"

#include <cassert>
#include <sstream>

#include <scm_core/console.h>
#include <scm_core/exception/system_exception.h>

using namespace scm::font;

face::face()
  : _size_at_72dpi(0),
    _line_spacing(0)
{
}

face::~face()
{
    clear();
}

bool face::has_style(face::style_type style) const
{
    face_style_mapping::const_iterator style_it = _glyph_mappings.find(style);

    if (style_it == _glyph_mappings.end()) {
        return (false);
    }

    return (true);
}

const std::string& face::get_name() const
{
    return (_name);
}

unsigned face::get_size() const
{
    return (_size_at_72dpi);
}

int face::get_line_spacing() const
{
    return (_line_spacing);
}

const glyph& face::get_glyph(unsigned char ind,
                             face::style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    character_glyph_mapping::const_iterator glyph_it = fstyle._glyph_mapping.find(ind);

    assert(glyph_it != fstyle._glyph_mapping.end());

    return (glyph_it->second);
}

int face::get_kerning(unsigned char    left,
                      unsigned char    right,
                      face::style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    return (fstyle._kerning_table[left][right]);
}

const face::face_style& face::get_face_style(style_type style) const
{
    face_style_mapping::const_iterator style_it = _glyph_mappings.find(style);

    if (style_it == _glyph_mappings.end()) {
        style_it = _glyph_mappings.find(face::regular);

        if (style_it == _glyph_mappings.end()) {
            std::stringstream output;

            output << "face::get_glyph(): "
                   << "unable to retrieve requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << std::endl;

            console.get() << con::log_level(con::error)
                          << output.str();

            throw scm::core::system_exception(output.str());
        }
    }

    return (style_it->second);
}

void face::clear()
{
    _glyph_mappings.clear();
    _name.assign("");
    _size_at_72dpi = 0;
    _line_spacing  = 0;
}
