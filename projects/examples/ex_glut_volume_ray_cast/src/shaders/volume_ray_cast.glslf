
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

vec3 light_pos   = vec3(10.0, 10.0, 0.0);
vec3 light_color = vec3(1.0, 1.0, 1.0);


vec3 get_gradient(vec3 sampling_pos, float f_distance, vec3 obj_to_tex)
{

    //float s_ohl = texture(volume_texture, (sampling_pos + vec3(f_distance, f_distance, f_distance   )) * obj_to_tex).r;
    //float s_ohr = texture(volume_texture, (sampling_pos + vec3(f_distance, f_distance, 0.0          )) * obj_to_tex).r;
    //float s_ovl = texture(volume_texture, (sampling_pos + vec3(f_distance, 0.0,        f_distance   )) * obj_to_tex).r;
    //float s_ovr = texture(volume_texture, (sampling_pos + vec3(f_distance, 0.0,        0.0          )) * obj_to_tex).r;
    //float s_uhl = texture(volume_texture, (sampling_pos + vec3(0.0,        f_distance, f_distance   )) * obj_to_tex).r;
    //float s_uhr = texture(volume_texture, (sampling_pos + vec3(0.0,        f_distance, 0.0          )) * obj_to_tex).r;
    //float s_uvl = texture(volume_texture, (sampling_pos + vec3(0.0,        0.0,        f_distance   )) * obj_to_tex).r;
    //float s_uvr = texture(volume_texture, (sampling_pos + vec3(0.0,        0.0,        0.0          )) * obj_to_tex).r;

    float grad_x = texture(volume_texture, (sampling_pos + vec3(f_distance, 0.0, 0.0 )) * obj_to_tex).r - texture(volume_texture, (sampling_pos - vec3(f_distance, 0, 0     )) * obj_to_tex).r;
    float grad_y = texture(volume_texture, (sampling_pos + vec3(0.0, f_distance, 0.0 )) * obj_to_tex).r - texture(volume_texture, (sampling_pos - vec3(0.0, f_distance, 0.0 )) * obj_to_tex).r;
    float grad_z = texture(volume_texture, (sampling_pos + vec3(0.0, 0.0, f_distance )) * obj_to_tex).r - texture(volume_texture, (sampling_pos - vec3(0.0, 0.0, f_distance )) * obj_to_tex).r;
    
    
    return(vec3(grad_x, grad_y, grad_z) * 0.5f);
}

vec4 
get_shading(in vec3 spos, in vec3 grad, in vec4 color)
{
    //vec3 col = vec3(0.0, 0.0, 0.0);
    
    //vec3 lt = (vec4(light_pos, 0.0) * volume_data.mv_matrix).rgb;
    vec3 lt = light_pos;

    vec3 n = normalize(grad);
    vec3 l = normalize(lt - spos);

    vec3 e = camera_location.xyz;
    vec3 v = normalize(e - spos);
    float ks = 0.8;

    vec3 r = (2 * n * dot(n, l)) - l;
        
    float color_shini = ks * pow(dot(r,v), 3);

    color.rgb = color.rgb * 0.3 + color.a * color.rgb * light_color * (dot(n, l) * 0.5f) + color_shini;
    //color.rgb =  light_color * (dot(n, l) * 0.5f + 0.5f);
    
    //return (vec4(col, color.a));
    return color;
}

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
        

        // increment ray
        sampling_pos  += ray_increment;
        inside_volume  = inside_volume_bounds(sampling_pos) && (dst.a < 0.99);
        
        
        if(s > 0.3){

            vec4 color = vec4(vec3(0.0), 1.0);

            vec3 gradient = get_gradient(sampling_pos, 2.0*sampling_distance, obj_to_tex);
            if(length(gradient) > 0.1){
                color = get_shading(sampling_pos, gradient, color);
            }

           dst = color;
        }
    }

#elif 0
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
