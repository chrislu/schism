
#ifndef FONT_FACE_H_INCLUDED
#define FONT_FACE_H_INCLUDED

#include <map>
#include <string>

#include <boost/multi_array.hpp>

#include <scm/core/font/glyph.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

class face_loader;

class __scm_export(core) face
{
public:
    typedef enum {
        regular     = 0x01,
        italic,
        bold,
        bold_italic
    } style_type;

protected:
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

    int                             get_line_spacing() const;

    bool                            has_style(style_type /*style*/) const;

    const glyph&                    get_glyph(unsigned char /*ind*/,
                                              style_type    /*style*/ = regular) const;
    int                             get_kerning(unsigned char /*left*/,
                                                unsigned char /*right*/,
                                                style_type    /*style*/ = regular) const;

protected:
    const face_style&               get_face_style(style_type /*style*/) const;

private:
    std::string                     _name;
    unsigned                        _size_at_72dpi;
    unsigned                        _line_spacing;

    void                            clear();

    face_style_mapping              _glyph_mappings;

    friend class face_loader;
}; // class face

} // namespace font
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // FONT_FACE_H_INCLUDED
