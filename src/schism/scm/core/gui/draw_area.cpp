
#include "draw_area.h"

using namespace scm;
using namespace scm::gui;

draw_area::draw_area()
  : _position(0),
    _size(0),
    _opacity(1.f)
{
}

draw_area::~draw_area()
{
}

void draw_area::update()
{
}

void draw_area::position(const math::vec2i_t& pos)
{
    _position = pos;
}

const math::vec2i_t& draw_area::position() const
{
    return (_position);
}

void draw_area::size(const math::vec2i_t& s)
{
    _size = s;
}

const math::vec2i_t& draw_area::size() const
{
    return (_size);
}

void draw_area::opacity(float op)
{
    _opacity = op;
}

float draw_area::opacity() const
{
    return (_opacity);
}
