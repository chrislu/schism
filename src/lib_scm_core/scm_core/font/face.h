
#ifndef FONT_FACE_H_INCLUDED
#define FONT_FACE_H_INCLUDED

#include <map>
#include <string>

#include <boost/multi_array.hpp>

#include <scm_core/font/glyph.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

class __scm_export face
{
public:
    typedef enum {
        regular     = 0x01,
        italic,
        bold,
        bold_italic
    } style_type;

    typedef std::map<unsigned, glyph>               character_glyph_mapping;
    typedef boost::multi_array<char, 2>             kerning_table;

    typedef struct {
        character_glyph_mapping     _glyph_mapping;
        kerning_table               _kerning_table;
    } face_style;

    typedef std::map<style_type, face_style>        face_style_mapping;

public:
    face();
    virtual ~face();

    const std::string&              get_name() const;
    unsigned                        get_size() const;

    bool                            has_style(style_type   /*style*/) const;

    const glyph&                    get_glyph(style_type   /*style*/ = regular) const;
    int                             get_kerning(unsigned   /*left*/,
                                                unsigned   /*right*/,
                                                style_type /*style*/ = regular) const;

protected:

private:
    std::string                     _name;
    unsigned                        _size_at_72dpi; // pixel_size... rough but nearest

    character_glyph_mapping         _glyph_mapping;

}; // class face

} // namespace font
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // FONT_FACE_H_INCLUDED
