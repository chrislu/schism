
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 420 core

uniform mat4 mvp;

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_tex_coord;

out per_vertex {
    smooth vec2 texcoord;
} v_out;

void main() {
    gl_Position    = mvp * vec4(in_position, 1.0);
    v_out.texcoord = in_tex_coord;
}
