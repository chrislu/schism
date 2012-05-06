
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 420 core

#extension GL_ARB_shading_language_include : require

#if SCM_TEXT_NV_BINDLESS_TEXTURES == 1
#extension GL_NV_bindless_texture : require
#endif

#include </scm/gl_util/camera_block.glslh>

// attribute layout definitions ///////////////////////////////////////////////////////////////////
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

// input/output definitions ///////////////////////////////////////////////////////////////////////
out per_vertex {
    smooth vec3 ray_entry_os;
    smooth vec3 ray_entry_ts;
} vertex_out;

// uniform input definitions //////////////////////////////////////////////////////////////////////
layout(std140, column_major) uniform;

uniform volume_uniform_data
{
    vec4 volume_extends;     // w unused
    vec4 scale_obj_to_tex;   // w unused
    vec4 sampling_distance;  // yzw unused
    vec4 os_camera_position;
    vec4 value_range;        // vec4f(min_value(), max_value(), max_value() - min_value(), 1.0f / (max_value() - min_value()));

    mat4 m_matrix;
    mat4 m_matrix_inverse;
    mat4 m_matrix_inverse_transpose;

    mat4 mv_matrix;
    mat4 mv_matrix_inverse;
    mat4 mv_matrix_inverse_transpose;

    mat4 mvp_matrix;
    mat4 mvp_matrix_inverse;

#if SCM_TEXT_NV_BINDLESS_TEXTURES == 1
    sampler3D volume_texture;
    sampler1D color_map;
#endif // SCM_TEXT_NV_BINDLESS_TEXTURES == 1
} volume_data;

// implementation /////////////////////////////////////////////////////////////////////////////////
void main()
{
    vertex_out.ray_entry_os = in_position;
    vertex_out.ray_entry_ts = in_position;
    gl_Position             = volume_data.mvp_matrix * vec4(in_position, 1.0);
    //gl_Position           = camera_transform.vp_matrix * vec4(in_position, 1.0);
}
