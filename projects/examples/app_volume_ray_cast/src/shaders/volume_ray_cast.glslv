
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 330 core

out vec3 ray_entry_position;

//uniform mat4 projection_matrix;
//uniform mat4 model_view_matrix;
//uniform mat4 model_view_matrix_inverse;
//uniform mat4 model_view_matrix_inverse_transpose;

layout(std140, column_major) uniform;

uniform transform_matrices
{
    mat4 mv_matrix;
    mat4 mv_matrix_inverse;
    mat4 mv_matrix_inverse_transpose;

    mat4 p_matrix;
    mat4 p_matrix_inverse;

    mat4 mvp_matrix;
    mat4 mvp_matrix_inverse;
} current_transform;

layout(location = 0) in vec3 in_position;

void main()
{
    //vec4 cam_pos = model_view_matrix_inverse * vec4(0.0, 0.0, 0.0, 1.0);

    //camera_position    = cam_pos.xyz / cam_pos.w;
    ray_entry_position = in_position;

    gl_Position = current_transform.mvp_matrix * vec4(in_position, 1.0);
}
