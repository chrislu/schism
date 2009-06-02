
#include "box.h"

#include <algorithm>

namespace scm {
namespace gl {

box::box(const box::vec3_type& min_vert,
         const box::vec3_type& max_vert)
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

const box::vec3_type&
box::min_vertex() const
{
    return (_min_vertex);
}

const box::vec3_type&
box::max_vertex() const
{
    return (_max_vertex);
}

const box::vec3_type
box::center() const
{
    return ((_max_vertex + _min_vertex) * 0.5f);
}

const box::vec3_type
box::corner(unsigned ind) const
{
    return (vec3_type(ind & 1 ? _max_vertex.x : _min_vertex.x,
                      ind & 2 ? _max_vertex.y : _min_vertex.y,
                      ind & 4 ? _max_vertex.z : _min_vertex.z));
}

void
box::min_vertex(const box::vec3_type& vert)
{
    _min_vertex = vert;
}

void
box::max_vertex(const box::vec3_type& vert)
{
    _max_vertex = vert;
}

} // namespace gl
} // namespace scm
