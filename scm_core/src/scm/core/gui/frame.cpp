
#include "frame.h"

namespace scm {
namespace gui {

frame::frame()
  : _content_margins(0, 0, 0, 0),
    _background_mode(frame::background_mode_none),
    _background_color(0.f, 0.f, 0.f, 1.f)
{
}

frame::~frame()
{
}

void frame::background_mode(frame::background_mode_t mode)
{
    _background_mode = mode;
}

frame::background_mode_t frame::background_mode() const
{
    return (_background_mode);
}

void frame::background_color(const scm::math::vec4f& col)
{
    _background_color = col;
}

const scm::math::vec4f& frame::background_color() const
{
    return (_background_color);
}

void frame::content_margins(const scm::math::vec4i& margins)
{
    _content_margins = margins;
}

const scm::math::vec4i& frame::content_margins() const
{
    return (_content_margins);
}

} // namespace gui
} // namespace scm
