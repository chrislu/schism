
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PRIMITIVES_RECT_H_INCLUDED
#define SCM_GL_CORE_PRIMITIVES_RECT_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/gl_core/primitives/primitives_fwd.h>
#include <scm/gl_core/primitives/plane.h>
#include <scm/gl_core/primitives/ray.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

template<typename s>
class rect_impl
{
public:
    typedef scm::math::vec<s, 2>    vec2_type;
    typedef scm::math::vec<s, 3>    vec3_type;
    typedef ray_impl<s>             ray_type;
    typedef plane_impl<s>           plane_type;

    typedef enum {
        inside,
        outside,
        overlaping
    } classification_result;

public:
    rect_impl(const vec2_type& min_vert = vec2_type(0.0f),
              const vec2_type& max_vert = vec2_type(1.0f));
    rect_impl(const rect_impl& b);
    virtual ~rect_impl();

    rect_impl&                  operator=(const rect_impl& b);
    void                        swap(rect_impl& b);

    const vec3_type&            min_vertex() const;
    const vec3_type&            max_vertex() const;
    const vec3_type             center() const;
    const vec3_type             corner(unsigned ind) const;

    void                        min_vertex(const vec2_type& vert);
    void                        max_vertex(const vec2_type& vert);

    const plane_type&           poly_plane() const;

    classification_result       classify(const vec3_type& p) const;
    classification_result       classify(const rect_impl& a) const;

    bool                        intersect(const ray_type& r,
                                          vec3_type&      hit) const;

protected:
    vec3_type                   _min_vertex;
    vec3_type                   _max_vertex;

    plane_type                  _poly_plane;

}; //class box

} // namespace gl
} // namespace scm

namespace std {

template<typename s>
inline void swap(scm::gl::rect_impl<s>& lhs,
                 scm::gl::rect_impl<s>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include "rect.inl"
#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_PRIMITIVES_RECT_H_INCLUDED
