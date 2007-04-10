
#ifndef SCM_MATH_H_INCLUDED
#define SCM_MATH_H_INCLUDED

#include <scm_core/math/math_lib.h>
#include <scm_core/math/math_vec.h>
#include <scm_core/math/math_vec_lib.h>
#include <scm_core/math/math_mat.h>
#include <scm_core/math/math_mat_lib.h>

namespace math
{
    typedef float               scalf_t;

    typedef double              scald_t;

    typedef vec<float, 2>       vec2f_t;
    typedef vec<float, 3>       vec3f_t;
    typedef vec<float, 4>       vec4f_t;

    typedef vec<double, 2>      vec2d_t;
    typedef vec<double, 3>      vec3d_t;
    typedef vec<double, 4>      vec4d_t;

    typedef vec<int, 2>         point_t;

    typedef mat<float, 2, 2>    mat2x2f_t;
    typedef mat<float, 3, 3>    mat3x3f_t;
    typedef mat<float, 4, 4>    mat4x4f_t;

    typedef mat<double, 2, 2>   mat2x2d_t;
    typedef mat<double, 3, 3>   mat3x3d_t;
    typedef mat<double, 4, 4>   mat4x4d_t;

    // some constants
    const scalf_t               pi_f = scalf_t(3.14159265358979323846264338327950288);
    const scald_t               pi_d = scald_t(3.14159265358979323846264338327950288);

    const mat<float, 4, 4>      mat4x4f_identity = mat<float, 4, 4>(1.0f, 0.0f, 0.0f, 0.0f,
                                                                    0.0f, 1.0f, 0.0f, 0.0f,
                                                                    0.0f, 0.0f, 1.0f, 0.0f,
                                                                    0.0f, 0.0f, 0.0f, 1.0f);


} // math

#endif // SCM_MATH_H_INCLUDED


