
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PRIMITIVES_RAY_H_INCLUDED
#define SCM_GL_CORE_PRIMITIVES_RAY_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

template <typename s>
class ray_impl
{
public:
    typedef scm::math::vec<s, 4>    vec4_type;
    typedef scm::math::vec<s, 3>    vec3_type;
    typedef scm::math::mat<s, 4, 4> mat4_type;

public:
    ray_impl();
    ray_impl(const ray_impl& p);
    explicit ray_impl(const vec3_type& org,
                      const vec3_type& dir);

    ray_impl&               operator=(const ray_impl& rhs);
    void                    swap(ray_impl& rhs);

    void                    transform(const mat4_type& t);
    void                    transform_preinverted(const mat4_type& t);

    const vec3_type&        origin() const;
    const vec3_type&        direction() const;

protected:
    void                    normalize();

protected:
    vec3_type               _origin;
    vec3_type               _direction;

}; // class ray

} // namespace gl
} // namespace scm

namespace std {

template<typename s>
inline void swap(scm::gl::ray_impl<s>& lhs,
                 scm::gl::ray_impl<s>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include "ray.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_PRIMITIVES_RAY_H_INCLUDED
