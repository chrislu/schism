
#include "frame.h"

using namespace scm;
using namespace scm::gui;

frame::frame()
  : _content_margins(0),
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

void frame::background_color(const math::vec4f_t& col)
{
    _background_color = col;
}

const math::vec4f_t& frame::background_color() const
{
    return (_background_color);
}

void frame::content_margins(const math::vec4i_t& margins)
{
    _content_margins = margins;
}

const math::vec4i_t& frame::content_margins() const
{
    return (_content_margins);
}
