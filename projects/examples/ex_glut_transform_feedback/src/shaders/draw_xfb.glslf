
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 330 core

in per_vertex {
    smooth vec3 position;
    smooth vec3 normal;
} out_vertex;

uniform vec3    light_ambient;
uniform vec3    light_diffuse;
uniform vec3    light_specular;
uniform vec3    light_position;

uniform vec3    material_ambient;
uniform vec3    material_diffuse;
uniform vec3    material_specular;
uniform float   material_shininess;
uniform float   material_opacity;

uniform sampler2D color_texture_aniso;
uniform sampler2D color_texture_nearest;

layout(location = 0, index = 0) out vec4        out_color;

void main()
{
    vec4 res;
    vec3 n = normalize(out_vertex.normal);
    vec3 l = normalize(light_position); // assume parallel light!

    vec4 c;
    c.rgb = material_diffuse;//texture(color_texture_aniso, texture_coord_fs);

    res.rgb =  light_ambient * material_ambient
             + light_diffuse * c.rgb/*material_diffuse*/ * max(0.0, dot(n, l));

    res.a = material_opacity;

    out_color = vec4(n, 1.0);//res;//
}

