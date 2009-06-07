
#include "ray.h"

#include <algorithm>

namespace scm {
namespace gl {

ray::ray()
  : _origin(ray::vec3_type::value_type(0)),
    _direction(ray::vec3_type::value_type(0))
{
}

ray::ray(const ray& p)
  : _origin(p._origin),
    _direction(p._direction)
{
}

ray::ray(const ray::vec3_type& org,
         const ray::vec3_type& dir)
  : _origin(org),
    _direction(dir)
{
    normalize();
}

ray&
ray::operator=(const ray& rhs)
{
    ray tmp(rhs);
    swap(tmp);
    return (*this);
}

void
ray::swap(ray& rhs)
{
    std::swap(_origin,    rhs._origin);
    std::swap(_direction, rhs._direction);
}

void
ray::transform(const ray::mat4_type& t)
{
    using namespace scm::math;

    mat4_type inv_trans = transpose(inverse(t));

    _direction = vec3_type(inv_trans * vec4_type(_direction, vec4_type::value_type(0)));
    _origin    = vec3_type(t         * vec4_type(_origin,    vec4_type::value_type(1)));

    normalize();
}

void
ray::transform_preinverted(const ray::mat4_type& it)
{
    using namespace scm::math;

    mat4_type inv_trans = transpose(it);

    _direction = vec3_type(inv_trans   * vec4_type(_direction, vec4_type::value_type(0)));
    _origin    = vec3_type(inverse(it) * vec4_type(_origin,    vec4_type::value_type(1)));

    normalize();
}

const ray::vec3_type&
ray::origin() const
{
    return (_origin);
}

const ray::vec3_type&
ray::direction() const
{
    return (_direction);
}

void
ray::normalize()
{
    _direction = math::normalize(_direction);
}

} // namespace gl
} // namespace scm
