
#include "text_box.h"

using namespace scm;
using namespace scm::gui;

text_box::text_box()
  : _text_orientation(horizontal),
    _text_flow(top_to_bottom),
    _text_alignment(left)
{
}

text_box::~text_box()
{
}

void text_box::text_orientation(gui::text_orientation ori)
{
    _text_orientation = ori;
}

void text_box::text_flow(gui::text_flow flow)
{
    _text_flow = flow;
}

void text_box::text_alignment(gui::text_alignment align)
{
    _text_alignment = align;
}

text_orientation text_box::text_orientation() const
{
    return (_text_orientation);
}

text_flow text_box::text_flow() const
{
    return (_text_flow);
}

text_alignment text_box::text_alignment() const
{
    return (_text_alignment);
}
