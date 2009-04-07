
#include "face.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

#include <scm/log.h>

namespace scm {
namespace font {

// face_style
face_style::face_style()
{
}

face_style::~face_style()
{
    clear();
}

const glyph& face_style::get_glyph(unsigned char ind) const
{
    character_glyph_mapping::const_iterator glyph_it = _glyph_mapping.find(ind);

    assert(glyph_it != _glyph_mapping.end());

    return (glyph_it->second);
}

unsigned face_style::get_line_spacing() const
{
    return (_line_spacing);
}

int face_style::get_kerning(unsigned char left,
                            unsigned char right) const
{
    return (_kerning_table[left][right]);
}

void face_style::clear()
{
    _glyph_mapping.clear();
    _kerning_table.resize(boost::extents[0][0]);
}

int face_style::get_underline_position() const
{
    return (_underline_position);
}

int face_style::get_underline_thickness() const
{
    return (_underline_thickness);
}

// face
face::face()
  : _size_at_72dpi(0)
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

unsigned face::get_line_spacing(style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    return (fstyle.get_line_spacing());
}

const glyph& face::get_glyph(unsigned char ind,
                             face::style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    return (fstyle.get_glyph(ind));
}

int face::get_kerning(unsigned char    left,
                      unsigned char    right,
                      face::style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    return (fstyle.get_kerning(left, right));
}

int face::get_underline_position(style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    return (fstyle.get_underline_position());
}

int face::get_underline_thickness(style_type style) const
{
    const face_style&   fstyle = get_face_style(style);

    return (fstyle.get_underline_thickness());
}

const face_style& face::get_face_style(style_type style) const
{
    face_style_mapping::const_iterator style_it = _glyph_mappings.find(style);

    if (style_it == _glyph_mappings.end()) {
        style_it = _glyph_mappings.find(face::regular);

        if (style_it == _glyph_mappings.end()) {
            std::stringstream output;

            output << "scm::font::face::get_glyph(): "
                   << "unable to retrieve requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << std::endl;

            scm::err() << scm::log_level(scm::logging::ll_error)
                       << output.str();

            throw std::runtime_error(output.str());
        }
    }

    return (style_it->second);
}

void face::clear()
{
    _glyph_mappings.clear();
    _name.assign("");
    _size_at_72dpi = 0;
}

} // namespace font
} // namespace scm
