
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 330 core

in vec3 ray_entry_position;

uniform sampler3D volume_texture;
uniform sampler1D color_map_texture;

uniform vec3    camera_location;
uniform float   sampling_distance;
uniform vec3    max_bounds;

layout(location = 0) out vec4        out_color;

vec3 debug_col;

bool
inside_volume_bounds(const in vec3 sampling_position)
{
    return (   all(greaterThanEqual(sampling_position, vec3(0.0)))
            && all(lessThanEqual(sampling_position, max_bounds)));
}

void main()
{
    vec3 ray_increment      = normalize(ray_entry_position - camera_location) * sampling_distance;
    vec3 sampling_pos       = ray_entry_position + ray_increment; // test, increment just to be sure we are in the volume

    vec3 obj_to_tex         = vec3(1.0) / max_bounds;

    vec4 dst = vec4(0.0, 0.0, 0.0, 0.0);

    bool inside_volume = inside_volume_bounds(sampling_pos);

#if 1
    while (inside_volume) {
        // get sample
        float s = texture(volume_texture, sampling_pos * obj_to_tex).r;
        vec4 src = texture(color_map_texture, s);

        // increment ray
        sampling_pos  += ray_increment;
        inside_volume  = inside_volume_bounds(sampling_pos) && (dst.a < 0.99);
        // compositing
        float omda_sa = (1.0 - dst.a) * src.a;
        dst.rgb += omda_sa*src.rgb;
        dst.a   += omda_sa;
    }
#elif 0
    bool inside_volume = true;
    int loop_c = 0;
    while (true) {
        loop_c += 1;

        src = texture(volume_texture, sampling_pos * obj_to_tex);
        src = texture(color_map_texture, src.r);

        sampling_pos  += ray_increment;
        inside_volume = (loop_c < 1000);
        // compositing
        float omda_sa = (1.0 - dst.a) * src.a;
        dst.rgb += omda_sa*src.rgb;
        dst.a   += omda_sa;
    }
#else
    for (int lc = 0; lc < 1000; ++lc) {
        src = texture(volume_texture, sampling_pos * obj_to_tex);
        src = texture(color_map_texture, src.r);

        sampling_pos  += ray_increment;

        // compositing
        float omda_sa = (1.0 - dst.a) * src.a;
        dst.rgb += omda_sa*src.rgb;
        dst.a   += omda_sa;
    }
#endif

    //vec4 volume_col = texture(volume_texture, ray_entry_position * 0.5);
    //vec4 color_map  = texture(color_map_texture, volume_col.r);

    //out_color = vec4(volume_col.rgb, 1.0);
    //out_color = vec4(color_map.rgb, 1.0);
    //out_color = vec4(ray_entry_position, 1.0);
    //out_color = vec4(sampling_pos, 1.0);
    out_color = dst;
}
