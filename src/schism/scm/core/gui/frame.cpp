
#include "frame.h"

using namespace scm;
using namespace scm::gui;

frame::frame()
  : _content_margins(0)
{
}

frame::~frame()
{
}

void frame::content_margins(const math::vec4i_t& margins)
{
    _content_margins = margins;
}

const math::vec4i_t& frame::content_margins() const
{
    return (_content_margins);
}
