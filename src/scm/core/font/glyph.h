
#ifndef FONT_GLYPH_H_INCLUDED
#define FONT_GLYPH_H_INCLUDED

#include <string>

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

struct __scm_export(core) glyph
{
    scm::math::vec2i    _tex_lower_left;
    scm::math::vec2i    _tex_upper_right;

    unsigned            _advance;

    scm::math::vec2i    _bearing;

}; // struct glyph

} // namespace font
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // FONT_GLYPH_H_INCLUDED
