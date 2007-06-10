
#include "font_renderer_2d.h"

#include <scm/core/utilities/foreach.h>

#include <scm/ogl/gl.h>
#include <scm/ogl/textures/texture_2d_rect.h>

namespace {

void draw_quad(const math::vec2i_t& lower_left,
               const math::vec2i_t& upper_right,
               const math::vec2i_t& tex_lower_left,
               const math::vec2i_t& tex_upper_right)
{
    glBegin(GL_QUADS);
        glTexCoord2i(tex_lower_left.x,  tex_lower_left.y);
        glVertex2i(  lower_left.x,      lower_left.y);

        glTexCoord2i(tex_upper_right.x, tex_lower_left.y);
        glVertex2i(  upper_right.x,     lower_left.y);

        glTexCoord2i(tex_upper_right.x, tex_upper_right.y);
        glVertex2i(  upper_right.x,     upper_right.y);

        glTexCoord2i(tex_lower_left.x,  tex_upper_right.y);
        glVertex2i(  lower_left.x,      upper_right.y);
    glEnd();
}

} // namespace

using namespace scm;
using namespace scm::gl;

font_renderer_2d::font_renderer_2d()
  : _shadow_color(0.f),
    _draw_shadow(true),
    _use_kering(true)
{
}

font_renderer_2d::~font_renderer_2d()
{
}

void font_renderer_2d::active_font(const scm::gl::font_face& font)
{
    _active_font    = font;
}

const font_face& font_renderer_2d::active_font() const
{
    return (_active_font);
}

unsigned font_renderer_2d::get_current_line_advance() const
{
    return (_active_font.get().get_line_spacing());
}

unsigned font_renderer_2d::calculate_text_width(const std::string&      txt,
                                                font::face::style_type  stl) const
{
    if (!_active_font) {
        return (0);
    }

    const scm::font::face_style&    cur_style     = _active_font.get().get_face_style(stl);

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

void font_renderer_2d::draw_shadow(bool sh)
{
    _draw_shadow = sh;
}

bool font_renderer_2d::draw_shadow() const
{
    return (_draw_shadow);
}

void font_renderer_2d::use_kerning(bool k)
{
    _use_kering = k;
}

bool font_renderer_2d::use_kerning() const
{
    return (_use_kering);
}

void font_renderer_2d::draw_string(const math::vec2i_t&         pos,
                                   const std::string&           txt,
                                   bool                         unl,
                                   scm::font::face::style_type  stl) const
{
    return (draw_string(pos, txt, math::vec4f_t(1.f, 1.f, 1.f, 1.f), unl, stl));
}

void font_renderer_2d::draw_string(const math::vec2i_t&         pos,
                                   const std::string&           txt,
                                   const math::vec3f_t          col,
                                   bool                         unl,
                                   scm::font::face::style_type  stl) const
{
    return (draw_string(pos, txt, math::vec4f_t(col, 1.f), unl, stl));
}

void font_renderer_2d::draw_string(const math::vec2i_t&         pos,
                                   const std::string&           txt,
                                   const math::vec4f_t          col,
                                   bool                         unl,
                                   scm::font::face::style_type  stl) const
{
    if (!_active_font) {
        return;
    }

    const scm::gl::texture_2d_rect& cur_style_tex = _active_font.get().get_glyph_texture(stl);
    const scm::font::face_style&    cur_style     = _active_font.get().get_face_style(stl);

    // save states which we change in here
    glPushAttrib(  GL_LIGHTING_BIT
                 | GL_DEPTH_BUFFER_BIT
                 | GL_COLOR_BUFFER_BIT
                 | GL_LINE_BIT);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    // setup blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    cur_style_tex.bind();

    math::vec2i_t current_pos = pos;

    unsigned char               prev        = 0;
    const math::vec2i_t         shadow_off  = math::vec2i_t(1,-1);

    foreach (unsigned char c, txt) {
        const scm::font::glyph& cur_glyph = cur_style.get_glyph(c);

        // kerning
        if (_use_kering && prev) {
            current_pos.x += cur_style.get_kerning(prev, c);
        }

        // draw shadow first
        if (_draw_shadow) {
            glColor4f(_shadow_color.x, _shadow_color.y, _shadow_color.z, col.w);
            draw_quad(current_pos + shadow_off + cur_glyph._bearing,
                      current_pos + shadow_off + cur_glyph._bearing + cur_glyph._tex_upper_right - cur_glyph._tex_lower_left,
                      cur_glyph._tex_lower_left,
                      cur_glyph._tex_upper_right);
        }

        // draw glyph
        glColor4fv(col.vec_array);
        draw_quad(current_pos + cur_glyph._bearing,
                  current_pos + cur_glyph._bearing + cur_glyph._tex_upper_right - cur_glyph._tex_lower_left,
                  cur_glyph._tex_lower_left,
                  cur_glyph._tex_upper_right);

        // advance the position
        current_pos.x += cur_glyph._advance;

        // remember just drawn glyph for kerning
        prev = c;
    }
    
    cur_style_tex.unbind();
    
    if (unl) {
        glLineWidth(static_cast<float>(cur_style.get_underline_thickness()));
        // draw underline shadow first
        if (_draw_shadow) {
            glColor4f(_shadow_color.x, _shadow_color.y, _shadow_color.z, col.w);
            glBegin(GL_LINES);
                glVertex2i(pos.x + shadow_off.x,
                           pos.y + shadow_off.y + cur_style.get_underline_position());
                glVertex2i(current_pos.x + shadow_off.x,
                           current_pos.y + shadow_off.y + cur_style.get_underline_position());
            glEnd();
        }

        // draw underline
        glColor4fv(col.vec_array);
        glBegin(GL_LINES);
            glVertex2i(pos.x, pos.y + cur_style.get_underline_position());
            glVertex2i(current_pos.x, pos.y + cur_style.get_underline_position());
        glEnd();
    }

    // restore saved states
    glPopAttrib();
}
