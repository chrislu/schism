
#include "face.h"

using namespace scm::font;

face::face()
{
}

face::~face()
{
}

bool face::has_style(face::style_type style) const
{
    return (false);
}

const std::string& face::get_name() const
{
    return (_name);
}

unsigned face::get_size() const
{
    return (_size_at_72dpi);
}

const glyph& face::get_glyph(face::style_type style) const
{
    //// HAAAACK
    character_glyph_mapping::const_iterator glyph_it = _glyph_mapping.find(0);
    return (glyph_it->second);
}

int face::get_kerning(unsigned         left,
                      unsigned         right,
                      face::style_type style) const
{
    return (0);
}
