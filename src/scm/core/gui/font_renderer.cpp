
#include "font_renderer.h"

#include <scm/core/utilities/foreach.h>

namespace {
} // namespace

namespace scm {
namespace gui {

font_renderer::font_renderer()
  : _shadow_color(0.f),
    _draw_shadow(true),
    _use_kering(true)
{
}

font_renderer::~font_renderer()
{
}

void font_renderer::active_font(const font::face_ptr& font)
{
    _active_font    = font;
}

const font::face_ptr& font_renderer::active_font() const
{
    return (_active_font);
}

unsigned font_renderer::get_current_line_advance() const
{
    if (!_active_font) {
        return (0);
    }

    return (_active_font->get_line_spacing());
}

unsigned font_renderer::calculate_text_width(const std::string&      txt,
                                             font::face::style_type  stl) const
{
    if (!_active_font) {
        return (0);
    }

    const scm::font::face_style&    cur_style     = _active_font->get_face_style(stl);

    unsigned char   prev        = 0;
    unsigned        ret_width   = 0;

    foreach (unsigned char c, txt) {
        const scm::font::glyph& cur_glyph = cur_style.get_glyph(c);

        // kerning
        if (_use_kering && prev) {
            ret_width += cur_style.get_kerning(prev, c);
        }
        // advance the position
        ret_width += cur_glyph._advance;
    }

    return (ret_width);
}

void font_renderer::draw_shadow(bool sh)
{
    _draw_shadow = sh;
}

bool font_renderer::draw_shadow() const
{
    return (_draw_shadow);
}

void font_renderer::use_kerning(bool k)
{
    _use_kering = k;
}

bool font_renderer::use_kerning() const
{
    return (_use_kering);
}

void font_renderer::draw_string(const math::vec2i_t&         pos,
                                const std::string&           txt,
                                bool                         unl,
                                scm::font::face::style_type  stl) const
{
    return (draw_string(pos, txt, math::vec4f_t(1.f, 1.f, 1.f, 1.f), unl, stl));
}

void font_renderer::draw_string(const math::vec2i_t&         pos,
                                const std::string&           txt,
                                const math::vec3f_t          col,
                                bool                         unl,
                                scm::font::face::style_type  stl) const
{
    return (draw_string(pos, txt, math::vec4f_t(col, 1.f), unl, stl));
}

} // namespace gui
} // namespace scm
