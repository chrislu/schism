
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PRIMITIVES_FRUSTUM_H_INCLUDED
#define SCM_GL_CORE_PRIMITIVES_FRUSTUM_H_INCLUDED

#include <vector>

#include <scm/core/math/math.h>

#include <scm/gl_core/primitives/primitives_fwd.h>
#include <scm/gl_core/primitives/plane.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

template<typename s>
class frustum_impl
{
public:
    typedef scm::math::mat<s, 4, 4> mat4_type;
    typedef box_impl<s>             box_type;
    typedef rect_impl<s>            rect_type;
    typedef plane_impl<s>           plane_type;

public:
    typedef enum {
        left_plane        = 0x00,
        right_plane       = 0x01,
        top_plane         = 0x02,
        bottom_plane      = 0x03,
        near_plane        = 0x04,
        far_plane         = 0x05
    } plane_identifier;

    typedef enum {
        inside,
        outside,
        intersecting
    } classification_result;

public:
    frustum_impl(const mat4_type& mvp_matrix = mat4_type::identity());
    frustum_impl(const frustum_impl& f);

    frustum_impl&           operator=(const frustum_impl& rhs);
    void                    swap(frustum_impl& rhs);

    void                    update(const mat4_type& mvp_matrix);
    const plane_type&       get_plane(unsigned int p) const;

    void                    transform(const mat4_type& t);
    void                    transform_preinverted(const mat4_type& t);

    classification_result   classify(const box_type& b) const;
    classification_result   classify(const rect_type& b) const;

protected:
    std::vector<plane_type> _planes;

}; // class frustum

} // namespace gl
} // namespace scm

namespace std {

template<typename s>
inline void swap(scm::gl::frustum_impl<s>& lhs,
                 scm::gl::frustum_impl<s>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include "frustum.inl"
#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_PRIMITIVES_FRUSTUM_H_INCLUDED
