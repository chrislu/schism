
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 430 core

#extension GL_NV_bindless_texture : require
#extension GL_NV_gpu_shader5      : require

// input layout definitions ///////////////////////////////////////////////////////////////////////

// input/output definitions ///////////////////////////////////////////////////////////////////////
in per_vertex {
    vec3 normal;
    vec2 texcoord;
    vec3 view_dir;
} v_in;

// attribute layout definitions ///////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;

// uniform input definitions //////////////////////////////////////////////////////////////////////
layout(binding  = 0) uniform sampler2D  tex_color;
layout(binding  = 1) uniform usampler2D tex_color_view;
uniform uvec2        tex_color_resident_lin;
uniform uvec2        tex_color_resident_near;

// global constants ///////////////////////////////////////////////////////////////////////////////
const float epsilon                 = 0.0001;

const vec3 white = vec3(1.0, 1.0, 1.0);
const vec3 black = vec3(0.0, 0.0, 0.0);
const vec3 red   = vec3(1.0, 0.0, 0.0);
const vec3 green = vec3(0.0, 1.0, 0.0);
const vec3 blue  = vec3(0.0, 0.0, 1.0);
const vec3 lblue = vec3(0.2, 0.7, 0.9);

// implementation /////////////////////////////////////////////////////////////////////////////////
void main() 
{
    vec4 c = vec4(0.0);
    if (gl_FragCoord.x < 400) {
        c = vec4(texture(tex_color_view, v_in.texcoord)) / 255.0;
        c *= vec4(0.5, 0.5, 1.0, 1.0);
    }
    else if (gl_FragCoord.x < 800) {
        c = texture(tex_color, v_in.texcoord);
    }
    else if (gl_FragCoord.x < 1200) {
        uint64_t  tex_hndl = packUint2x32(tex_color_resident_near);
        sampler2D tex_smpl = sampler2D(tex_hndl);
        c = texture(tex_smpl, v_in.texcoord);

        c *= vec4(0.5, 1.0, 0.5, 1.0);
    }
    else {
        uint64_t  tex_hndl = packUint2x32(tex_color_resident_lin);
        sampler2D tex_smpl = sampler2D(tex_hndl);
        c = texture(tex_smpl, v_in.texcoord);

        c *= vec4(1.0, 0.5, 0.5, 1.0);
    }

    vec3 n = normalize(v_in.normal); 
    vec3 l = normalize(vec3(1.0, 1.0, 1.0)); // assume parallel light!
    vec3 v = normalize(v_in.view_dir);
    vec3 h = normalize(l + v);

    out_color.rgb =    c.rgb * (dot(n, l) * 0.5 + 0.5)//max(0.0, dot(n, l))
                     + lblue * pow(max(0.0, dot(n, h)), 120.0);
    
    out_color.rgb =  + c.rgb;
    out_color.a   = 1.0;
}

