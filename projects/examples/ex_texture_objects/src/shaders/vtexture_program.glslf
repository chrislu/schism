
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 410 core

#extension GL_ARB_shading_language_include : require

#include </scm/data/vtexture/vtexture.glslh>
#include </scm/data/vtexture/vtexture_debug.glslh>

// input layout definitions ///////////////////////////////////////////////////////////////////////
layout(early_fragment_tests) in;

// input/output definitions ///////////////////////////////////////////////////////////////////////
in per_vertex {
    vec3 normal;
    vec2 texcoord;
    vec3 view_dir;
} v_in;

// attribute layout definitions ///////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;

// uniform input definitions //////////////////////////////////////////////////////////////////////
uniform vtexture2D  vtex_color;

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
    vtexture_require(vtex_color, v_in.texcoord);
    //vec4 c = draw_debug_quadtree();//
    vec4 c = vtexture(vtex_color, v_in.texcoord);//out_color_gen();//vec4(v_in.normal, 1.0);
    //vec4 c = vtexture_debug_page_trilinear(vtex_color, v_in.texcoord);//out_color_gen();//vec4(v_in.normal, 1.0);
    //vec4 c = vtexture_debug_quadtree(vtex_color, v_in.texcoord);//out_color_gen();//vec4(v_in.normal, 1.0);
    //vec4 c = vtexture_debug_atlas(vtex_color, v_in.texcoord);//out_color_gen();//vec4(v_in.normal, 1.0);

    vec3 n = normalize(v_in.normal); 
    vec3 l = normalize(vec3(1.0, 1.0, 1.0)); // assume parallel light!
    vec3 v = normalize(v_in.view_dir);
    vec3 h = normalize(l + v);

    out_color.rgb =    c.rgb * (dot(n, l) * 0.5 + 0.5)//max(0.0, dot(n, l))
                     + lblue * pow(max(0.0, dot(n, h)), 120.0);

    //out_color.rgb =  + c.rgb;
    out_color.a   = 1.0;
}

