
#include "frustum.h"

namespace scm {
namespace gl {

frustum::frustum(const scm::math::mat4f& mvp_matrix)
{
    update(mvp_matrix);
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

    _planes[left_plane]   = plane(tmp_plane / length(vec3f(tmp_plane)));

    // right plane
    tmp_plane.x = mvp_matrix.m03 - mvp_matrix.m00;
    tmp_plane.y = mvp_matrix.m07 - mvp_matrix.m04;
    tmp_plane.z = mvp_matrix.m11 - mvp_matrix.m08;
    tmp_plane.w = mvp_matrix.m15 - mvp_matrix.m12;

    _planes[right_plane]  = plane(tmp_plane / length(vec3f(tmp_plane)));

    // bottom plane
    tmp_plane.x = mvp_matrix.m03 + mvp_matrix.m01;
    tmp_plane.y = mvp_matrix.m07 + mvp_matrix.m05;
    tmp_plane.z = mvp_matrix.m11 + mvp_matrix.m09;
    tmp_plane.w = mvp_matrix.m15 + mvp_matrix.m13;

    _planes[bottom_plane]    = plane(tmp_plane / length(vec3f(tmp_plane)));

    // top plane
    tmp_plane.x = mvp_matrix.m03 - mvp_matrix.m01;
    tmp_plane.y = mvp_matrix.m07 - mvp_matrix.m05;
    tmp_plane.z = mvp_matrix.m11 - mvp_matrix.m09;
    tmp_plane.w = mvp_matrix.m15 - mvp_matrix.m13;

    _planes[top_plane] = plane(tmp_plane / length(vec3f(tmp_plane)));

    // near plane
    tmp_plane.x = mvp_matrix.m03 + mvp_matrix.m02;
    tmp_plane.y = mvp_matrix.m07 + mvp_matrix.m06;
    tmp_plane.z = mvp_matrix.m11 + mvp_matrix.m10;
    tmp_plane.w = mvp_matrix.m15 + mvp_matrix.m14;

    _planes[near_plane]   = plane(tmp_plane / length(vec3f(tmp_plane)));

    // far plane
    tmp_plane.x = mvp_matrix.m03 - mvp_matrix.m02;
    tmp_plane.y = mvp_matrix.m07 - mvp_matrix.m06;
    tmp_plane.z = mvp_matrix.m11 - mvp_matrix.m10;
    tmp_plane.w = mvp_matrix.m15 - mvp_matrix.m14;

    _planes[far_plane]    = plane(tmp_plane / length(vec3f(tmp_plane)));
}

const plane&
frustum::get_plane(unsigned int p) const
{
    return (_planes[p]);
}

} // namespace gl
} // namespace scm
