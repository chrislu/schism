
#ifndef FONT_GLYPH_H_INCLUDED
#define FONT_GLYPH_H_INCLUDED

#include <string>

#include <scm_core/math/math.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

struct __scm_export glyph
{
    math::vec2i_t       _tex_lower_left;
    math::vec2i_t       _tex_upper_right;

    unsigned            _advance;

    math::vec2i_t       _bearing;

}; // struct glyph

} // namespace font
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // FONT_GLYPH_H_INCLUDED
