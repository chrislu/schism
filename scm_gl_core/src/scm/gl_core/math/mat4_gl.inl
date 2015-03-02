
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <scm/core/math/common.h>
//#include <scm/gl/opengl.h>

namespace scm {
namespace math {

template<typename scal_type>
inline void translate(mat<scal_type, 4, 4>&    m,
                      const vec<scal_type, 3>& t)
{
    m.m12 = m.m00 * t.x + m.m04 * t.y + m.m08 * t.z + m.m12;
    m.m13 = m.m01 * t.x + m.m05 * t.y + m.m09 * t.z + m.m13;
    m.m14 = m.m02 * t.x + m.m06 * t.y + m.m10 * t.z + m.m14;
    m.m15 = m.m03 * t.x + m.m07 * t.y + m.m11 * t.z + m.m15;
}

template<typename scal_type>
inline void translate(mat<scal_type, 4, 4>&    m,
                      const scal_type          x,
                      const scal_type          y,
                      const scal_type          z)
{
    translate(m, vec<scal_type, 3>(x, y, z));
}

template<typename scal_type>
inline void rotate(mat<scal_type, 4, 4>&       m,
                   const scal_type             angl,
                   const vec<scal_type, 3>&    axis)
{
    // code after mesagl example
    if (scm::math::length(axis) == scal_type(0)) {
        return;
    }

    mat<scal_type, 4, 4>    tmp_rot;
    scal_type               s = scm::math::sin(scm::math::deg2rad(angl));
    scal_type               c = scm::math::cos(scm::math::deg2rad(angl));
    scal_type               one_c = scal_type(1) - c;
    scal_type               xx, yy, zz, xy, yz, zx, xs, ys, zs;

    math::vec<scal_type, 3> axis_n = scm::math::normalize(axis);

    xx = axis_n.x * axis_n.x;
    yy = axis_n.y * axis_n.y;
    zz = axis_n.z * axis_n.z;
    xy = axis_n.x * axis_n.y;
    yz = axis_n.y * axis_n.z;
    zx = axis_n.z * axis_n.x;
    xs = axis_n.x * s;
    ys = axis_n.y * s;
    zs = axis_n.z * s;

    tmp_rot.m00 = (one_c * xx) + c;
    tmp_rot.m04 = (one_c * xy) - zs;
    tmp_rot.m08 = (one_c * zx) + ys;
    tmp_rot.m12 = scal_type(0);

    tmp_rot.m01 = (one_c * xy) + zs;
    tmp_rot.m05 = (one_c * yy) + c;
    tmp_rot.m09 = (one_c * yz) - xs;
    tmp_rot.m13 = scal_type(0);

    tmp_rot.m02 = (one_c * zx) - ys;
    tmp_rot.m06 = (one_c * yz) + xs;
    tmp_rot.m10 = (one_c * zz) + c;
    tmp_rot.m14 = scal_type(0);

    tmp_rot.m03 = scal_type(0);
    tmp_rot.m07 = scal_type(0);
    tmp_rot.m11 = scal_type(0);
    tmp_rot.m15 = scal_type(1);

    m *= tmp_rot;
}

template<typename scal_type>
inline void rotate(mat<scal_type, 4, 4>&       m,
                   const scal_type             angl,
                   const scal_type             axis_x,
                   const scal_type             axis_y,
                   const scal_type             axis_z)
{
    rotate(m, angl, vec<scal_type, 3>(axis_x, axis_y, axis_z));
}

template<typename scal_type>
inline void scale(mat<scal_type, 4, 4>&        m,
                  const vec<scal_type, 3>&     s)
{
    m.m00 *= s.x;
    m.m05 *= s.y;
    m.m10 *= s.z;
}

template<typename scal_type>
inline void scale(mat<scal_type, 4, 4>&        m,
                  const scal_type              x,
                  const scal_type              y,
                  const scal_type              z)
{
    scale(m, vec<scal_type, 3>(x, y, z));
}

//inline void get_gl_matrix(const int type, mat<float, 4, 4>& m)
//{
//    glGetFloatv(type, m.data_array);
//}

//inline void get_gl_matrix(const int type, mat<double, 4, 4>& m)
//{
//    glGetDoublev(type, m.data_array);
//}

template<typename scal_type>
inline
void
ortho_matrix(mat<scal_type, 4, 4>& m,
             scal_type left, scal_type right,
             scal_type bottom, scal_type top,
             scal_type near_z, scal_type far_z)
{
    scal_type A,B,C,D,E,F;

    A=-(right+left)/(right-left);
    B=-(top+bottom)/(top-bottom);
    C=-(far_z+near_z)/(far_z-near_z);

    D=-2.0f/(far_z-near_z);
    E=2.0f/(top-bottom);
    F=2.0f/(right-left);

    m = mat<scal_type, 4, 4>( F, 0, 0, 0,
                              0, E, 0, 0,
                              0, 0, D, 0,
                              A, B, C, 1);
}

template<typename scal_type>
inline
void
frustum_matrix(mat<scal_type, 4, 4>& m,
               scal_type left, scal_type right,
               scal_type bottom, scal_type top,
               scal_type near_z, scal_type far_z)
{
    scal_type A,B,C,D,E,F;

    A=(right+left)/(right-left);
    B=(top+bottom)/(top-bottom);
    C=-(far_z+near_z)/(far_z-near_z);
    D=-(2.0f*far_z*near_z)/(far_z-near_z);
    E=2.0f*near_z/(top-bottom);
    F=2.0f*near_z/(right-left);

    m = mat<scal_type, 4, 4>(F, 0, 0, 0,
                             0, E, 0, 0,
                             A, B, C,-1,
                             0, 0, D, 0);
}

template<typename scal_type>
inline
void
perspective_matrix(mat<scal_type, 4, 4>& m,
                   scal_type fovy,
                   scal_type aspect,
                   scal_type near_z,
                   scal_type far_z)
{
    scal_type maxy = math::tan(deg2rad(fovy * scal_type(.5))) * near_z;
    scal_type maxx = maxy*aspect;

    frustum_matrix(m, -maxx, maxx, -maxy, maxy, near_z, far_z);
}

template<typename scal_type>
inline
void
look_at_matrix(mat<scal_type, 4, 4>& m,
               const vec<scal_type, 3>& eye,
               const vec<scal_type, 3>& center,
               const vec<scal_type, 3>& up)
{
    vec<scal_type, 3> z_axis = normalize(center - eye);
    vec<scal_type, 3> y_axis = normalize(up);
    vec<scal_type, 3> x_axis = normalize(cross(z_axis, y_axis));
    y_axis = normalize(cross(x_axis, z_axis));

    m.data_array[0]  =  x_axis.x;
    m.data_array[1]  =  y_axis.x;
    m.data_array[2]  = -z_axis.x;
    m.data_array[3]  = scal_type(0.0);

    m.data_array[4]  =  x_axis.y;
    m.data_array[5]  =  y_axis.y;
    m.data_array[6]  = -z_axis.y;
    m.data_array[7]  = scal_type(0.0);

    m.data_array[8]  =  x_axis.z;
    m.data_array[9]  =  y_axis.z;
    m.data_array[10] = -z_axis.z;
    m.data_array[11] = scal_type(0.0);

    m.data_array[12] = scal_type(0.0);
    m.data_array[13] = scal_type(0.0);
    m.data_array[14] = scal_type(0.0);
    m.data_array[15] = scal_type(1.0);

    translate(m, -eye);
}

template<typename scal_type>
inline
void
look_at_matrix_inv(mat<scal_type, 4, 4>& m,
                   const vec<scal_type, 3>& eye,
                   const vec<scal_type, 3>& center,
                   const vec<scal_type, 3>& up)
{
    vec<scal_type, 3> z_axis = normalize(center - eye);
    vec<scal_type, 3> y_axis = normalize(up);
    vec<scal_type, 3> x_axis = normalize(cross(z_axis, y_axis));
    y_axis = normalize(cross(x_axis, z_axis));

    m.data_array[0]  =  x_axis.x;
    m.data_array[1]  =  x_axis.y;
    m.data_array[2]  =  x_axis.z;
    m.data_array[3]  = scal_type(0.0);

    m.data_array[4]  =  y_axis.x;
    m.data_array[5]  =  y_axis.y;
    m.data_array[6]  =  y_axis.z;
    m.data_array[7]  = scal_type(0.0);

    m.data_array[8]  = -z_axis.x;
    m.data_array[9]  = -z_axis.y;
    m.data_array[10] = -z_axis.z;
    m.data_array[11] = scal_type(0.0);

    m.data_array[12] = eye.x;
    m.data_array[13] = eye.y;
    m.data_array[14] = eye.z;
    m.data_array[15] = scal_type(1.0);
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_translation(const vec<scal_type, 3>& t)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    translate(ret, t);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_translation(const scal_type x, const scal_type y, const scal_type z)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    translate(ret, x, y, z);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_rotation(const scal_type angl, const vec<scal_type, 3>& axis)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    rotate(ret, angl, axis);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_rotation(const scal_type angl, const scal_type axis_x, const scal_type axis_y, const scal_type axis_z)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    rotate(ret, angl, axis_x, axis_y, axis_z);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_scale(const vec<scal_type, 3>& s)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    scale(ret, s);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_scale(const scal_type x, const scal_type y, const scal_type z)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    scale(ret, x, y, z);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_ortho_matrix(scal_type left, scal_type right, scal_type bottom, scal_type top, scal_type near_z, scal_type far_z)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    ortho_matrix(ret, left, right, bottom, top, near_z, far_z);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_frustum_matrix(scal_type left, scal_type right, scal_type bottom, scal_type top, scal_type near_z, scal_type far_z)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    frustum_matrix(ret, left, right, bottom, top, near_z, far_z);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_perspective_matrix(scal_type fovy, scal_type aspect, scal_type near_z, scal_type far_z)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    perspective_matrix(ret, fovy, aspect, near_z, far_z);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_look_at_matrix(const vec<scal_type, 3>& eye, const vec<scal_type, 3>& center, const vec<scal_type, 3>& up)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    look_at_matrix(ret, eye, center, up);

    return ret;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
make_look_at_matrix_inv(const vec<scal_type, 3>& eye, const vec<scal_type, 3>& center, const vec<scal_type, 3>& up)
{
  mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::identity();

    look_at_matrix_inv(ret, eye, center, up);

    return ret;
}

} // namespace math
} // namespace scm
