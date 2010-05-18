
#include "text_box.h"

#include <vector>


#include <scm/core/utilities/foreach.h>
#include <scm/gl_classic/gui/font_renderer.h>

namespace scm {
namespace gl_classic {
namespace gui {

text_box::text_box()
{
    _font_renderer.reset(new gl_classic::gui::font_renderer());
}

text_box::~text_box()
{
}

void text_box::draw()
{
    //scm::gl_classic::gui::frame::draw();
    draw_text();
}

} // namespace gui
} // namespace gl_classic
} // namespace scm
