
#include "plane.h"

namespace scm {
namespace gl {

plane::plane()
  : _vector(scm::math::vec4f(0.0f))
{
}

plane::plane(const plane& p)
  : _vector(p._vector)
{
}

plane::plane(const scm::math::vec4f& p)
  : _vector(p)
{
}

plane&
plane::operator=(const plane& rhs)
{
    _vector = rhs._vector;

    return (*this);
}

const scm::math::vec3f
plane::normal() const
{
    return (scm::math::vec3f(_vector));
}

scm::math::vec3f::value_type
plane::distance() const
{
    return (_vector.w);
}

const scm::math::vec4f&
plane::vector() const
{
    return (_vector);
}

} // namespace gl
} // namespace scm
