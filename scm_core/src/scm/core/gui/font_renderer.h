
#ifndef SCM_GUI_FONT_RENDERER_2D_H_INCLUDED
#define SCM_GUI_FONT_RENDERER_2D_H_INCLUDED

#include <scm/core/math/math.h>
#include <scm/core/font/face.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class __scm_export(core) font_renderer
{
public:
    font_renderer();
    virtual ~font_renderer();

    void                    active_font(const font::face_ptr& /*font*/);
    const font::face_ptr&   active_font() const;

    unsigned                get_current_line_advance() const;
    unsigned                calculate_text_width(const std::string&      /*txt*/,
                                                 font::face::style_type  /*stl*/ = font::face::regular) const;

    void                    draw_shadow(bool /*sh*/);
    bool                    draw_shadow() const;

    void                    use_kerning(bool /*k*/);
    bool                    use_kerning() const;

    virtual void            draw_string(const scm::math::vec2i& /*pos*/,
                                        const std::string&      /*txt*/,
                                        bool                    /*unl*/ = false,
                                        font::face::style_type  /*stl*/ = font::face::regular) const;
    virtual void            draw_string(const scm::math::vec2i& /*pos*/,
                                        const std::string&      /*txt*/,
                                        const scm::math::vec3f  /*col*/,
                                        bool                    /*unl*/ = false,
                                        font::face::style_type  /*stl*/ = font::face::regular) const;
    virtual void            draw_string(const scm::math::vec2i& /*pos*/,
                                        const std::string&      /*txt*/,
                                        const scm::math::vec4f& /*col*/,
                                        bool                    /*unl*/ = false,
                                        font::face::style_type  /*stl*/ = font::face::regular) const = 0;

protected:
    font::face_ptr      _active_font;
    bool                _draw_shadow;
    scm::math::vec3f    _shadow_color;
    bool                _use_kerning;

private:

}; // class font_renderer

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_GUI_FONT_RENDERER_2D_H_INCLUDED
