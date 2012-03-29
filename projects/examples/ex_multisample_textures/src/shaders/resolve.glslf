
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 420 core

layout(location = 0, index = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2DMS in_texture;

uniform int num_samples;

in per_vertex {
    smooth vec2 texcoord;
} v_in;

void main() {
    vec2  tex_size    = vec2(textureSize(in_texture));
    ivec2 texel_pos   = ivec2(v_in.texcoord * tex_size);
    vec4  accum_color = vec4(0.0);
    for(int s = 0; s < num_samples; ++s) {
        accum_color += texelFetch(in_texture, texel_pos, s);
    }

    out_color = accum_color / float(num_samples);
}
