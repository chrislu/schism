
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_texture_coord; 

uniform mat4 mvp;

noperspective out vec2 tex_coord;

void main()
{
    gl_Position = mvp * vec4(in_position, 1.0);
    tex_coord   = in_texture_coord;

    //gl_Position.xy = in_position.xy;
    //gl_Position.y *= -1.0;
    //gl_Position.zw = vec2(0.0, 1.0);

    //tex_coord = in_position.xy * 0.5 + 0.5;
}
