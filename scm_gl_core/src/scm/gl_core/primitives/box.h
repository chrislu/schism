
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PRIMITIVES_BOX_H_INCLUDED
#define SCM_GL_CORE_PRIMITIVES_BOX_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/gl_core/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

template<typename s>
class box_impl
{
public:
    typedef scm::math::vec<s, 3>    vec3_type;
    typedef ray_impl<s>             ray_type;

    typedef enum {
        inside,
        outside,
        overlaping
    } classification_result;

public:
    box_impl(const vec3_type& min_vert = vec3_type(0.0f),
             const vec3_type& max_vert = vec3_type(1.0f));
    box_impl(const box_impl& b);
    virtual ~box_impl();

    box_impl&                   operator=(const box_impl& b);
    void                        swap(box_impl& b);

    const vec3_type&            min_vertex() const;
    const vec3_type&            max_vertex() const;
    const vec3_type             center() const;
    const vec3_type             corner(unsigned ind) const;

    void                        min_vertex(const vec3_type& vert);
    void                        max_vertex(const vec3_type& vert);

    classification_result       classify(const vec3_type& p) const;
    classification_result       classify(const box_impl& a) const;

    bool                        intersect(const ray_type& r, vec3_type& entry, vec3_type& exit) const;

protected:
    vec3_type                   _min_vertex;
    vec3_type                   _max_vertex;

}; //class box

} // namespace gl
} // namespace scm

namespace std {

template<typename s>
inline void swap(scm::gl::box_impl<s>& lhs,
                 scm::gl::box_impl<s>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include "box.inl"
#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_PRIMITIVES_BOX_H_INCLUDED
