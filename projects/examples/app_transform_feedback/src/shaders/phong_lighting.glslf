
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 330 core

in per_vertex {
    smooth vec3 position;
    smooth vec3 normal;
    smooth vec2 texcoord;
    smooth vec3 view_dir;
} in_interpolant;

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

layout(location = 0) out vec4        out_color;

void main()
{
    vec4 res;
    vec3 n = normalize(in_interpolant.normal);
    vec3 l = normalize(light_position); // assume parallel light!
    vec3 v = normalize(in_interpolant.view_dir);
    vec3 h = normalize(l + v);//gl_LightSource[0].halfVector);//

    vec4 c;

#if 0
    if (texture_coord_fs.x > 0.5) {
        c = texture(color_texture_aniso, in_interpolant.texcoord);
    }
    else {
        c = texture(color_texture_nearest, in_interpolant.texcoord);
    }
#else
    c.rgb = material_diffuse;//texture(color_texture_aniso, texture_coord_fs);
#endif

#if 1
    res.rgb =  light_ambient * material_ambient
         + light_diffuse * c.rgb/*material_diffuse*/ * max(0.0, dot(n, l))
         + light_specular * material_specular * pow(max(0.0, dot(n, h)), material_shininess);
#else
    res.rgb = c.rgb;//vec3(texture_coord_fs, 0.0);//
#endif
    res.a = material_opacity;

    out_color = res;
}

