
#include "box.h"

namespace scm {
namespace gl {

box::box(const scm::math::vec3f& min_vert,
         const scm::math::vec3f& max_vert)
  : _min_vertex(min_vert),
    _max_vertex(max_vert)
{
}

box::~box()
{
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
box::centroid() const
{
    return ((_max_vertex - _min_vertex) * 0.5f);
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
