
#ifndef SCM_LARGE_DATA_VOLUME_UNIFORM_DATA_H_INCLUDED
#define SCM_LARGE_DATA_VOLUME_UNIFORM_DATA_H_INCLUDED

#ifndef __OPENCL_VERSION__
#include <scm/core/math.h>

typedef scm::math::vec4f    float4;
typedef scm::math::mat4f    float4x4;

#else

#if 0
struct mat4f
{
    float4 col[4];
};

typedef struct mat4f float4x4;

#else

typedef float16 float4x4;

#endif

#endif

struct volume_uniform_data
{
    float4      _volume_extends;     // w unused
    float4      _scale_obj_to_tex;   // w unused
    float4      _sampling_distance;  // yzw unused
    float4      _os_camera_position;
    float4      _value_range;
    
    // pad to get the matrices to 64B boundaries
    float4      _dummy[3];

    float4x4     _m_matrix;
    float4x4     _m_matrix_inverse;
    float4x4     _m_matrix_inverse_transpose;

    float4x4     _mv_matrix;
    float4x4     _mv_matrix_inverse;
    float4x4     _mv_matrix_inverse_transpose;

    float4x4     _mvp_matrix;
    float4x4     _mvp_matrix_inverse;
}; // struct volume_uniform_data

#endif // SCM_LARGE_DATA_VOLUME_UNIFORM_DATA_H_INCLUDED
