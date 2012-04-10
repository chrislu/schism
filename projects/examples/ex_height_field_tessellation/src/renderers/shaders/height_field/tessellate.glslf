
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 410 core

#extension GL_ARB_shading_language_include : require

#include </scm/gl_util/camera_block.glslh>

#include </scm/data/height_field/common_constants.glslh>
#include </scm/data/height_field/common_functions.glslh>
#include </scm/data/height_field/common_uniforms.glslh>
#include </scm/data/height_field/interface_blocks.glslh>

// output layout definitions //////////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;

// input/output definitions ///////////////////////////////////////////////////////////////////////
in per_vertex {
    smooth hf_vertex vertex;
} g_in;

// implementation /////////////////////////////////////////////////////////////////////////////////
void main()
{
#if 1
    float height_value = texture(height_map, g_in.vertex.texcoord_hm).r;
    vec4  mapped_color = texture(color_map, height_value);

    vec2 height_map_size       = vec2(textureSize(height_map, 0).xy);
    vec2 height_map_texel_size = vec2(1.0) / vec2(height_map_size);

    vec3 n_h = normalize(height_map_gradient(g_in.vertex.texcoord_hm, height_map_nearest, height_map_texel_size));
    vec3 n   = normalize(height_map_gradient(g_in.vertex.texcoord_hm, height_map, height_map_texel_size));

    if (dot(n_h, vec3(0.0, 0.0, 1.0)) < 0.1) discard;

    vec3 l = normalize(vec3(1.0, 1.0, 1.0)); // assume parallel light!
    vec3 v = normalize(camera_transform.ws_position.xyz - g_in.vertex.ws_position.xyz);
    vec3 h = normalize(l + v);

    //out_color.rgb =  vec3(0.95);//
    out_color.rgb =  + mapped_color.rgb * (dot(n, l) * 0.5 + 0.5)//max(0.0, dot(n, l))
                     + white * pow(max(0.0, dot(n, h)), 60.0);
#else
    //out_color.rgb = texture(height_map, g_in.vertex.texcoord_hm).rrr;
    out_color.rgb = texture(density_map, g_in.vertex.texcoord_dm).rrr;
    //out_color.rgb = pow(texture(height_map, vertex_in.texcoord).rrr, vec3(2.2));//vec3(1.0, 0.0, 0.0);//
    //out_color.rgb = texture(density_map, vertex_in.texcoord_dm).rrr;//vec3(1.0, 0.0, 0.0);//
    //out_color.rgb = 20.0 * texture(density_map, vertex_in.texcoord_dm).rrr;//vec3(1.0, 0.0, 0.0);//
    //out_color.rgb  = vec3(g_in.vertex.texcoord_hm.xy, 0.0);
    //out_color.rgb  = vertex_out.ws_position.xyz;
    //out_color.rgb = vec3(0.95);
#endif
    out_color.a   = 1.0;//0.25;
}
