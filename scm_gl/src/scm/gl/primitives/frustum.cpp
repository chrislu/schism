
#include "frustum.h"

#include <algorithm>

#include <scm/gl/primitives/box.h>

namespace scm {
namespace gl {

frustum::frustum(const scm::math::mat4f& mvp_matrix)
  : _planes(6)
{
    update(mvp_matrix);
}

frustum::frustum(const frustum& f)
  : _planes(f._planes)
{
}

frustum&
frustum::operator=(const frustum& rhs)
{
    frustum tmp(rhs);
    swap(tmp);
    return (*this);
}

void
frustum::swap(frustum& rhs)
{
    std::swap(_planes, rhs._planes);
}

void
frustum::update(const scm::math::mat4f& mvp_matrix)
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

const plane&
frustum::get_plane(unsigned int p) const
{
    return (_planes[p]);
}

void
frustum::transform(const frustum::mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted(inverse(t));
}

void
frustum::transform_preinverted(const frustum::mat4_type& t)
{
    using namespace scm::math;
    frustum::mat4_type inv_trans = transpose(t);
    for (unsigned i = 0; i < 6; ++i) {
        _planes[i].transform_preinverted_transposed(inv_trans);
    }
}

frustum::classification_result
frustum::classify(const box& b) const
{
    bool plane_intersect = false;

    // normals of frustum planes point inside the frustum
    for (unsigned i = 0; i < 6; ++i) {
        plane::classification_result cur_plane_res = _planes[i].classify(b);

        if (cur_plane_res == plane::back) {
            return (outside);
        }
        else if (cur_plane_res == plane::intersect) {
            plane_intersect = true;
        }
    }

    return (plane_intersect ? intersect : inside);
}

} // namespace gl
} // namespace scm
