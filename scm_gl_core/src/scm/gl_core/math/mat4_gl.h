
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_MAT4_GL_H_INCLUDED
#define SCM_GL_CORE_MAT4_GL_H_INCLUDED

#include <scm/core/math/mat.h>

namespace scm {
namespace math {

template<typename scal_type> void       translate(mat<scal_type, 4, 4>& m, const vec<scal_type, 3>& t);
template<typename scal_type> void       translate(mat<scal_type, 4, 4>& m, const scal_type x, const scal_type y, const scal_type z);
template<typename scal_type> void       rotate(mat<scal_type, 4, 4>& m, const scal_type angl, const vec<scal_type, 3>& axis);
template<typename scal_type> void       rotate(mat<scal_type, 4, 4>& m, const scal_type angl, const scal_type axis_x, const scal_type axis_y, const scal_type axis_z);
template<typename scal_type> void       scale(mat<scal_type, 4, 4>& m, const vec<scal_type, 3>& s);
template<typename scal_type> void       scale(mat<scal_type, 4, 4>& m, const scal_type x, const scal_type y, const scal_type z);
template<typename scal_type> void       ortho_matrix(mat<scal_type, 4, 4>& m, scal_type left, scal_type right, scal_type bottom, scal_type top, scal_type near_z, scal_type far_z); 
template<typename scal_type> void       frustum_matrix(mat<scal_type, 4, 4>& m, scal_type left, scal_type right, scal_type bottom, scal_type top, scal_type near_z, scal_type far_z);
template<typename scal_type> void       perspective_matrix(mat<scal_type, 4, 4>& m, scal_type fovy = 45, scal_type aspect = 4.0f/3.0f, scal_type near_z = 0.1f, scal_type far_z  = 100);
template<typename scal_type> void       look_at_matrix(mat<scal_type, 4, 4>& m, const vec<scal_type, 3>& eye, const vec<scal_type, 3>& center, const vec<scal_type, 3>& up);
template<typename scal_type> void       look_at_matrix_inv(mat<scal_type, 4, 4>& m, const vec<scal_type, 3>& eye, const vec<scal_type, 3>& center, const vec<scal_type, 3>& up);

// convenience
template<typename scal_type> const mat<scal_type, 4, 4>     make_translation(const vec<scal_type, 3>& t);
template<typename scal_type> const mat<scal_type, 4, 4>     make_translation(const scal_type x, const scal_type y, const scal_type z);
template<typename scal_type> const mat<scal_type, 4, 4>     make_rotation(const scal_type angl, const vec<scal_type, 3>& axis);
template<typename scal_type> const mat<scal_type, 4, 4>     make_rotation(const scal_type angl, const scal_type axis_x, const scal_type axis_y, const scal_type axis_z);
template<typename scal_type> const mat<scal_type, 4, 4>     make_scale(const vec<scal_type, 3>& s);
template<typename scal_type> const mat<scal_type, 4, 4>     make_scale(const scal_type x, const scal_type y, const scal_type z);
template<typename scal_type> const mat<scal_type, 4, 4>     make_ortho_matrix(scal_type left, scal_type right, scal_type bottom, scal_type top, scal_type near_z, scal_type far_z); 
template<typename scal_type> const mat<scal_type, 4, 4>     make_frustum_matrix(scal_type left, scal_type right, scal_type bottom, scal_type top, scal_type near_z, scal_type far_z);
template<typename scal_type> const mat<scal_type, 4, 4>     make_perspective_matrix(scal_type fovy = 45, scal_type aspect = 4.0f/3.0f, scal_type near_z = 0.1f, scal_type far_z  = 100);
template<typename scal_type> const mat<scal_type, 4, 4>     make_look_at_matrix(const vec<scal_type, 3>& eye, const vec<scal_type, 3>& center, const vec<scal_type, 3>& up);
template<typename scal_type> const mat<scal_type, 4, 4>     make_look_at_matrix_inv(const vec<scal_type, 3>& eye, const vec<scal_type, 3>& center, const vec<scal_type, 3>& up);

//void get_gl_matrix(const int type, mat<float, 4, 4>& m);
//void get_gl_matrix(const int type, mat<double, 4, 4>& m);

} // namespace math
} // namespace scm

#include "mat4_gl.inl"

#endif // SCM_GL_CORE_MAT4_GL_H_INCLUDED
