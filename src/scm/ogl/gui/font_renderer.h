
#ifndef GL_FONT_RENDERER_2D_H_INCLUDED
#define GL_FONT_RENDERER_2D_H_INCLUDED

#include <scm/core/gui/font_renderer.h>
#include <scm/ogl/font/face.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace gui {

class __scm_export(ogl) font_renderer : public scm::gui::font_renderer
{
public:
    font_renderer();
    virtual ~font_renderer();

    using scm::gui::font_renderer::draw_string;

    void                draw_string(const scm::math::vec2i& /*pos*/,
                                    const std::string&      /*txt*/,
                                    const scm::math::vec4f& /*col*/,
                                    bool                    /*unl*/ = false,
                                    font::face::style_type  /*stl*/ = font::face::regular) const;

protected:

private:

}; // class font_renderer

} // namespace gui
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // GL_FONT_RENDERER_2D_H_INCLUDED
