
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 330 core

out vec3 normal;
out vec2 texture_coord;
out vec3 view_dir;

uniform mat4 projection_matrix;
uniform mat4 model_view_matrix;
uniform mat4 model_view_matrix_inverse_transpose;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texture_coord;

void main()
{
    normal        =  normalize(model_view_matrix_inverse_transpose * vec4(in_normal, 0.0)).xyz;
    view_dir      = -normalize(model_view_matrix * vec4(in_position, 1.0)).xyz;
    texture_coord = in_texture_coord;

    gl_Position = projection_matrix * model_view_matrix * vec4(in_position, 1.0);
}
