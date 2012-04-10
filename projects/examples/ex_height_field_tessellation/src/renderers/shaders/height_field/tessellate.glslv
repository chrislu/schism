
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 410 core

#extension GL_ARB_shading_language_include : require

#include </scm/gl_util/camera_block.glslh>

#include </scm/data/height_field/common_constants.glslh>
#include </scm/data/height_field/common_uniforms.glslh>
#include </scm/data/height_field/interface_blocks.glslh>

// attribute layout definitions ///////////////////////////////////////////////////////////////////
layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_texcoord_hm;
layout(location = 3) in vec2 in_texcoord_dm;

// input/output definitions ///////////////////////////////////////////////////////////////////////
out per_vertex {
    smooth hf_vertex vertex;
} v_out;

// implementation /////////////////////////////////////////////////////////////////////////////////
void main()
{
    vec4 hf_position         = vec4(in_position, 1.0);
    hf_position.z            = texture(height_map, in_texcoord_hm).r * height_scale;
    gl_Position              = hf_position;

    v_out.vertex.ws_position = model_matrix * hf_position;
    v_out.vertex.texcoord_hm = in_texcoord_hm;
    v_out.vertex.texcoord_dm = in_texcoord_dm;
}
