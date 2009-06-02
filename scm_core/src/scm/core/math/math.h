
#ifndef SCM_MATH_H_INCLUDED
#define SCM_MATH_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/math/config.h>

#include <scm/core/math/common.h>

#include <scm/core/math/vec.h>
#include <scm/core/math/vec2.h>
#include <scm/core/math/vec3.h>
#include <scm/core/math/vec4.h>

#include <scm/core/math/mat.h>
#include <scm/core/math/mat2.h>
#include <scm/core/math/mat3.h>
#include <scm/core/math/mat4.h>

#include <scm/core/math/vec_stream_io.h>
#include <scm/core/math/mat_stream_io.h>

namespace scm {
namespace math {

typedef vec<float, 2>       vec2f;
typedef vec<float, 3>       vec3f;
typedef vec<float, 4>       vec4f;

typedef vec<double, 2>      vec2d;
typedef vec<double, 3>      vec3d;
typedef vec<double, 4>      vec4d;

typedef vec<int, 2>         vec2i;
typedef vec<int, 3>         vec3i;
typedef vec<int, 4>         vec4i;

typedef vec<unsigned, 2>    vec2ui;
typedef vec<unsigned, 3>    vec3ui;
typedef vec<unsigned, 4>    vec4ui;

typedef mat<float, 2, 2>    mat2f;
typedef mat<float, 3, 3>    mat3f;
typedef mat<float, 4, 4>    mat4f;

typedef mat<double, 2, 2>   mat2d;
typedef mat<double, 3, 3>   mat3d;
typedef mat<double, 4, 4>   mat4d;

// some constants
const float                 pi_f = 3.14159265358979323846264338327950288f;
const double                pi_d = 3.14159265358979323846264338327950288;

#if SCM_CORE_MATH_FP_PRECISION == SCM_CORE_MATH_FP_PRECISION_SINGLE
typedef vec2f               vec2;
typedef vec3f               vec3;
typedef vec4f               vec4;

typedef mat2f               mat2;
typedef mat3f               mat3;
typedef mat4f               mat4;
#elif SCM_CORE_MATH_FP_PRECISION == SCM_CORE_MATH_FP_PRECISION_DOUBLE
typedef vec2d               vec2;
typedef vec3d               vec3;
typedef vec4d               vec4;

typedef mat2d               mat2;
typedef mat3d               mat3;
typedef mat4d               mat4;
#else
#error "SCM_CORE_MATH_FP_PRECISION not defined"
#endif // SCM_CORE_MATH_FP_PRECISION

} // namespace math
} // namespace scm

#endif // SCM_MATH_H_INCLUDED
