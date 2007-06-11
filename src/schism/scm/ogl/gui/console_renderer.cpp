
#include "console_renderer.h"

using namespace scm;
using namespace scm::gl;
using namespace scm::gl::gui;

console_renderer::console_renderer()
{
    hor_alignment(scm::gui::hor_align_left);
    flow(scm::gui::flow_bottom_to_top);
    content_margins(math::vec4i_t(10));
}

console_renderer::~console_renderer()
{
}

void console_renderer::update(const std::string&                update_buffer,
                              const con::console_out_stream&    stream_source)
{
    append_string(update_buffer);
}
