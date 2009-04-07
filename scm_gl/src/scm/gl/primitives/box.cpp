
#include "box.h"

#include <algorithm>

namespace scm {
namespace gl {

box::box(const scm::math::vec3f& min_vert,
         const scm::math::vec3f& max_vert)
  : _min_vertex(min_vert),
    _max_vertex(max_vert)
{
}

box::box(const box& b)
  : _min_vertex(b._min_vertex),
    _max_vertex(b._max_vertex)
{
}

box::~box()
{
}

box&
box::operator=(const box& b)
{
    box tmp(b);

    swap(tmp);

    return (*this);
}

void
box::swap(box& b)
{
    std::swap(_min_vertex, b._min_vertex);
    std::swap(_max_vertex, b._max_vertex);
}

const scm::math::vec3f&
box::min_vertex() const
{
    return (_min_vertex);
}

const scm::math::vec3f&
box::max_vertex() const
{
    return (_max_vertex);
}

const scm::math::vec3f
box::center() const
{
    return ((_max_vertex + _min_vertex) * 0.5f);
}

void
box::min_vertex(const scm::math::vec3f& vert)
{
    _min_vertex = vert;
}

void
box::max_vertex(const scm::math::vec3f& vert)
{
    _max_vertex = vert;
}

} // namespace gl
} // namespace scm

namespace std {

void swap(scm::gl::box& lhs,
          scm::gl::box& rhs)
{
    lhs.swap(rhs);
}

} // namespace std
