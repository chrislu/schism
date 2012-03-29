
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 330 core

out per_vertex {
    smooth vec3 position;
    smooth vec3 normal;
} out_vertex;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

void main()
{
    out_vertex.normal   = in_normal;
    out_vertex.position = in_position;

    gl_Position = vec4(in_position, 1.0);
}
