
#ifndef GL_FONT_RENDERER_2D_H_INCLUDED
#define GL_FONT_RENDERER_2D_H_INCLUDED

#include <scm/core/math/math.h>
#include <scm/ogl/font/font.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

enum {
    align_left,
    align_right,
    align_center,
    align_bottom,
    align_top
};

class __scm_export(ogl) font_renderer_2d
{
public:
    font_renderer_2d();
    virtual ~font_renderer_2d();

    void                active_font(const scm::gl::font_face& /*font*/);
    const font_face&    active_font() const;

    unsigned            get_current_line_advance() const;
    unsigned            calculate_text_width(const std::string&      /*txt*/,
                                             font::face::style_type  /*stl*/ = font::face::regular) const;

    void                draw_shadow(bool /*sh*/);
    bool                draw_shadow() const;

    void                use_kerning(bool /*k*/);
    bool                use_kerning() const;

    void                draw_string(const math::vec2i_t&    /*pos*/,
                                    const std::string&      /*txt*/,
                                    bool                    /*unl*/ = false,
                                    font::face::style_type  /*stl*/ = font::face::regular) const;
    void                draw_string(const math::vec2i_t&    /*pos*/,
                                    const std::string&      /*txt*/,
                                    const math::vec3f_t     /*col*/,
                                    bool                    /*unl*/ = false,
                                    font::face::style_type  /*stl*/ = font::face::regular) const;
    void                draw_string(const math::vec2i_t&    /*pos*/,
                                    const std::string&      /*txt*/,
                                    const math::vec4f_t     /*col*/,
                                    bool                    /*unl*/ = false,
                                    font::face::style_type  /*stl*/ = font::face::regular) const;

protected:
    font_face           _active_font;
    bool                _draw_shadow;
    math::vec3f_t       _shadow_color;
    bool                _use_kering;

private:

}; // class font_renderer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // GL_FONT_RENDERER_2D_H_INCLUDED
