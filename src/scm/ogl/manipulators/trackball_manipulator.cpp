
#include "trackball_manipulator.h"

#include <scm/ogl/gl.h>
#include <scm/core/math/math.h>

namespace scm {
namespace gl {

// trackball_manipulator implementation
trackball_manipulator::trackball_manipulator()
  : _radius(1.f),
    _dolly(0.f),
   _matrix(scm::math::mat4f::identity)
{
}

trackball_manipulator::~trackball_manipulator()
{
}

float trackball_manipulator::project_to_sphere(float x, float y) const
{

    // inspired by OpenSG code of the TrackballNavigator class
    float len_sqr = x*x + y*y;
    float len     = math::sqrt(len_sqr);

    // if point lies inside sphere map it to the sphere, if it lies 
    // outside map it to hyperbola (at radius/sqrt(2) sphere and
    // hyperbola intersect and so this is the decission point)
    if (len < _radius / math::sqrt(2.0f)) {
        return (math::sqrt(_radius * _radius - len_sqr));
    } else {
        // hyperbola z = r²/(2*d)
        return ((_radius*_radius) / (2.0f * len));
    }
}

void trackball_manipulator::rotation(float fx, float fy, float tx, float ty)
{
    using namespace scm::math;

    // test if fx - tx or fy - ty > float.epsilon
    vec3f   start(fx, fy, project_to_sphere(fx, fy));
    vec3f   end(tx, ty, project_to_sphere(tx, ty));

    // difference vector between start and endpoint on the sphere
    vec3f   diff(end - start);
    float   diff_len = length(diff);

    // get rotation axis as cross product of the two vecs
    vec3f  rot_axis(cross(start, end));

    // calculate rotaion angle (assume start and end vectors are of 
    // equal length) a = asin(d/2*r) --> result is in radians
    float           rot_angl = 2.0f * asin(clamp(diff_len/(2.0f * _radius), -1.0f, 1.0f));

    mat4f tmp(mat4f::identity);

    rotate(tmp, rad2deg(rot_angl), rot_axis);

    _matrix = tmp * _matrix;
}

void trackball_manipulator::translation(float x, float y)
{
    using namespace scm::math;

    float dolly_abs = fabs(_dolly);
    float near_dist = 1.f; // rough estimate, maybe construct this better
                           // from the projection matrix

    mat4f tmp(mat4f::identity);

    translate(tmp,
              x * (near_dist + dolly_abs),
              y * (near_dist + dolly_abs),
              0.f);

    _matrix = tmp * _matrix;
}

void trackball_manipulator::dolly(float y) {
    _dolly -= y;
}

void trackball_manipulator::apply_transform() const {
    glTranslatef(0,0,_dolly);
    glMultMatrixf(&_matrix);
}

const scm::math::mat4f trackball_manipulator::get_transform_matrix() const
{
    using namespace scm::math;

    mat4f tmp_ret(mat4f::identity);

    translate(tmp_ret, 0.f, 0.f, _dolly);

    tmp_ret *= _matrix;

    return (tmp_ret);
}

float trackball_manipulator::dolly() const
{
    return (_dolly);
}

} // namespace gl
} // namespace scm

