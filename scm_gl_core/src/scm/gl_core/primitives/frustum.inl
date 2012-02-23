
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

#include <scm/gl_core/primitives/box.h>
#include <scm/gl_core/primitives/rect.h>

namespace scm {
namespace gl {

template<typename s>
frustum_impl<s>::frustum_impl(const typename frustum_impl<s>::mat4_type& mvp_matrix)
  : _planes(6)
{
    update(mvp_matrix);
}

template<typename s>
frustum_impl<s>::frustum_impl(const frustum_impl<s>& f)
  : _planes(f._planes)
{
}

template<typename s>
frustum_impl<s>&
frustum_impl<s>::operator=(const frustum_impl<s>& rhs)
{
    frustum_impl<s> tmp(rhs);
    swap(tmp);
    return (*this);
}

template<typename s>
void
frustum_impl<s>::swap(frustum_impl<s>& rhs)
{
    std::swap(_planes, rhs._planes);
}

template<typename s>
void
frustum_impl<s>::update(const typename frustum_impl<s>::mat4_type& mvp_matrix)
{
    using namespace scm::math;

    vec4f tmp_plane;

    // left plane
    tmp_plane.x = mvp_matrix.m03 + mvp_matrix.m00;
    tmp_plane.y = mvp_matrix.m07 + mvp_matrix.m04;
    tmp_plane.z = mvp_matrix.m11 + mvp_matrix.m08;
    tmp_plane.w = mvp_matrix.m15 + mvp_matrix.m12;

    _planes[left_plane]     = plane(tmp_plane);

    // right plane
    tmp_plane.x = mvp_matrix.m03 - mvp_matrix.m00;
    tmp_plane.y = mvp_matrix.m07 - mvp_matrix.m04;
    tmp_plane.z = mvp_matrix.m11 - mvp_matrix.m08;
    tmp_plane.w = mvp_matrix.m15 - mvp_matrix.m12;

    _planes[right_plane]    = plane(tmp_plane);

    // bottom plane
    tmp_plane.x = mvp_matrix.m03 + mvp_matrix.m01;
    tmp_plane.y = mvp_matrix.m07 + mvp_matrix.m05;
    tmp_plane.z = mvp_matrix.m11 + mvp_matrix.m09;
    tmp_plane.w = mvp_matrix.m15 + mvp_matrix.m13;

    _planes[bottom_plane]   = plane(tmp_plane);

    // top plane
    tmp_plane.x = mvp_matrix.m03 - mvp_matrix.m01;
    tmp_plane.y = mvp_matrix.m07 - mvp_matrix.m05;
    tmp_plane.z = mvp_matrix.m11 - mvp_matrix.m09;
    tmp_plane.w = mvp_matrix.m15 - mvp_matrix.m13;

    _planes[top_plane]      = plane(tmp_plane);

    // near plane
    tmp_plane.x = mvp_matrix.m03 + mvp_matrix.m02;
    tmp_plane.y = mvp_matrix.m07 + mvp_matrix.m06;
    tmp_plane.z = mvp_matrix.m11 + mvp_matrix.m10;
    tmp_plane.w = mvp_matrix.m15 + mvp_matrix.m14;

    _planes[near_plane]     = plane(tmp_plane);

    // far plane
    tmp_plane.x = mvp_matrix.m03 - mvp_matrix.m02;
    tmp_plane.y = mvp_matrix.m07 - mvp_matrix.m06;
    tmp_plane.z = mvp_matrix.m11 - mvp_matrix.m10;
    tmp_plane.w = mvp_matrix.m15 - mvp_matrix.m14;

    _planes[far_plane]      = plane(tmp_plane);
}

template<typename s>
const typename frustum_impl<s>::plane_type&
frustum_impl<s>::get_plane(unsigned int p) const
{
    return (_planes[p]);
}

template<typename s>
void
frustum_impl<s>::transform(const typename frustum_impl<s>::mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted(inverse(t));
}

template<typename s>
void
frustum_impl<s>::transform_preinverted(const typename frustum_impl<s>::mat4_type& t)
{
    using namespace scm::math;
    frustum_impl::mat4_type inv_trans = transpose(t);
    for (unsigned i = 0; i < 6; ++i) {
        _planes[i].transform_preinverted_transposed(inv_trans);
    }
}

template<typename s>
typename frustum_impl<s>::classification_result
frustum_impl<s>::classify(const box_type& b) const
{
    bool plane_intersect = false;

    // normals of frustum_impl planes point inside the frustum_impl
    for (unsigned i = 0; i < 6; ++i) {
        plane::classification_result cur_plane_res = _planes[i].classify(b);

        if (cur_plane_res == plane::back) {
            return (outside);
        }
        else if (cur_plane_res == plane::intersecting) {
            plane_intersect = true;
        }
    }

    return (plane_intersect ? intersecting : inside);
}

template<typename s>
typename frustum_impl<s>::classification_result
frustum_impl<s>::classify(const rect_type& b) const
{
    bool plane_intersect = false;

    // normals of frustum_impl planes point inside the frustum_impl
    for (unsigned i = 0; i < 6; ++i) {
        plane::classification_result cur_plane_res = _planes[i].classify(b);

        if (cur_plane_res == plane::back) {
            return (outside);
        }
        else if (cur_plane_res == plane::intersecting) {
            plane_intersect = true;
        }
    }

    return (plane_intersect ? intersecting : inside);
}

} // namespace gl
} // namespace scm
