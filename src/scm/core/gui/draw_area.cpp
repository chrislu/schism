
#include "draw_area.h"

namespace scm {
namespace gui {

draw_area::draw_area()
  : _position(0, 0),
    _size(0, 0),
    _opacity(1.f)
{
}

draw_area::~draw_area()
{
}

void draw_area::update()
{
}

void draw_area::position(const scm::math::vec2i& pos)
{
    _position = pos;
}

const scm::math::vec2i& draw_area::position() const
{
    return (_position);
}

void draw_area::size(const scm::math::vec2i& s)
{
    _size = s;
}

const scm::math::vec2i& draw_area::size() const
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

} // namespace gui
} // namespace scm
