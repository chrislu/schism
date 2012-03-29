
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 420 core

out vec3 normal;
out vec2 texture_coord;
out vec3 view_dir;

uniform mat4 projection_matrix;
uniform mat4 model_view_matrix;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

out per_vertex {
    smooth vec3 os_position;
    smooth vec3 os_normal;
    smooth vec3 es_position;
    smooth vec3 es_normal;
    smooth vec2 texcoord;
} v_out;

void main()
{
    v_out.os_normal   = in_normal;
    v_out.os_position = in_position;
    v_out.es_normal   = (transpose(inverse(model_view_matrix)) * vec4(in_normal, 0.0)).xyz;
    v_out.es_position = (model_view_matrix * vec4(in_position, 1.0)).xyz;
    v_out.texcoord    = in_texcoord;
    gl_Position       = projection_matrix * model_view_matrix  * vec4(in_position, 1.0);
}
