
#include "font_renderer.h"

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

namespace scm {
namespace gl {
namespace gui {

font_renderer::font_renderer()
{
}

font_renderer::~font_renderer()
{
}

void font_renderer::draw_string(const math::vec2i_t&         pos,
                                const std::string&           txt,
                                const math::vec4f_t          col,
                                bool                         unl,
                                scm::font::face::style_type  stl) const
{
    if (!_active_font) {
        return;
    }

    const gl::face& act_f = *boost::static_pointer_cast<gl::face>(_active_font);


    const scm::gl::texture_2d_rect& cur_style_tex = act_f.get_glyph_texture(stl);
    const scm::font::face_style&    cur_style     = act_f.get_face_style(stl);

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

} // namespace gui
} // namespace gl
} // namespace scm
