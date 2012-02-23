
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PRIMITIVES_PLANE_H_INCLUDED
#define SCM_GL_CORE_PRIMITIVES_PLANE_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/gl_core/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

// describing plane equation
//  - a*x + b*y + c*z + d = 0
//  - so d gives you the distance of the origin related to the plane!
template <typename s>
class plane_impl
{
public:
    typedef s                       scal_type;
    typedef scm::math::vec<s, 4>    vec4_type;
    typedef scm::math::vec<s, 3>    vec3_type;
    typedef scm::math::mat<s, 4, 4> mat4_type;
    typedef box_impl<s>             box_type;
    typedef rect_impl<s>            rect_type;
    typedef ray_impl<s>             ray_type;

    typedef enum {
        front,
        back,
        coinciding,
        intersecting
    } classification_result;

public:
    plane_impl();
    plane_impl(const plane_impl& p);
    plane_impl(const vec3_type& p0, const vec3_type& p1, const vec3_type& p2);
    explicit plane_impl(const vec4_type& p);

    plane_impl&             operator=(const plane_impl& rhs);
    void                    swap(plane_impl& rhs);

    const vec3_type         normal() const;
    scal_type               distance() const;                       // distance of the origin to the plane(-d)
    scal_type               distance(const vec3_type& p) const;     // distance of a point to the plane

    const vec4_type&        vector() const;

    void                    reverse();

    void                    transform(const mat4_type& t);
    void                    transform_preinverted(const mat4_type& t);
    void                    transform_preinverted_transposed(const mat4_type& t);

    unsigned                p_corner() const;
    unsigned                n_corner() const;

    classification_result   classify(const box_type& b, scal_type e = epsilon<scal_type>::value()) const;
    classification_result   classify(const rect_type& b, scal_type e = epsilon<scal_type>::value()) const;
    classification_result   classify(const vec3_type& p, scal_type e = epsilon<scal_type>::value()) const;

    bool                    intersect(const ray_type& r, vec3_type& hit, scal_type e = epsilon<scal_type>::value()) const;

protected:
    void                    normalize();
    void                    update_corner_indices();

protected:
    vec4_type               _vector;

    unsigned                _p_corner;
    unsigned                _n_corner;

}; // class frustum

} // namespace gl
} // namespace scm

namespace std {

template<typename s>
inline void swap(scm::gl::plane_impl<s>& lhs,
                 scm::gl::plane_impl<s>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include "plane.inl"
#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_PRIMITIVES_PLANE_H_INCLUDED
