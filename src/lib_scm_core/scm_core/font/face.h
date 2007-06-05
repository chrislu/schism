
#ifndef FONT_FACE_H_INCLUDED
#define FONT_FACE_H_INCLUDED

#include <map>
#include <string>

#include <scm_core/font/glyph.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

class __scm_export face
{
public:
    typedef std::map<unsigned, glyph>       character_glyph_mapping;

public:
    face();
    virtual ~face();

    const std::string&              get_name() const;
    const unsigned                  get_size() const;

    const glyph&                    get_glyph() const;
    int                             get_kerning(unsigned left, unsigned right) const;

protected:
    std::string                     _name;
    unsigned                        _size_at_72dpi;

    character_glyph_mapping         _glyph_mapping;

private:

}; // class face

} // namespace font
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // FONT_FACE_H_INCLUDED
