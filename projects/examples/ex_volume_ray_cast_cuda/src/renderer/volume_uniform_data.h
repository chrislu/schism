
#ifndef SCM_LARGE_DATA_VOLUME_UNIFORM_DATA_H_INCLUDED
#define SCM_LARGE_DATA_VOLUME_UNIFORM_DATA_H_INCLUDED


#ifndef __CUDACC__
#include <scm/core/math.h>

typedef scm::math::vec4f    vec4;
typedef scm::math::mat4f    float4x4;

#else

#include <vector_types.h>

typedef float4 vec4;
struct float4x4
{
    float4 rows[4];
};

#endif

struct volume_uniform_data
{
    vec4         _volume_extends;     // w unused
    vec4         _scale_obj_to_tex;   // w unused
    vec4         _sampling_distance;  // yzw unused
    vec4         _os_camera_position;
    vec4         _value_range;

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
