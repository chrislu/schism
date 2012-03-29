
#version 420 core

#extension GL_ARB_shading_language_include : require

#include </scm/gl_util/camera_block.glsl>

// output layout definitions //////////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;


 precision highp float;

// global constants ///////////////////////////////////////////////////////////////////////////////
const float epsilon                 = 0.0001;

const vec3 white = vec3(1.0, 1.0, 1.0);
const vec3 black = vec3(0.0, 0.0, 0.0);
const vec3 red   = vec3(1.0, 0.0, 0.0);
const vec3 green = vec3(0.0, 1.0, 0.0);
const vec3 blue  = vec3(0.0, 0.0, 1.0);

// input/output definitions ///////////////////////////////////////////////////////////////////////
in per_vertex {
    smooth vec3 ray_entry_os;
    smooth vec3 ray_entry_ts;
} vertex_in;

// uniform input definitions //////////////////////////////////////////////////////////////////////
uniform sampler3D volume_raw;
uniform sampler2D color_map;

uniform vec2  viewport_size;
uniform float volume_lod;

layout(std140, column_major) uniform;

uniform volume_uniform_data
{
    vec4 volume_extends;     // w unused
    vec4 scale_obj_to_tex;   // w unused
    vec4 sampling_distance;  // x - os sampling distance, y opacity correction factor, zw unused
    vec4 os_camera_position;
    vec4 value_range;        // vec4f(min_value(), max_value(), max_value() - min_value(), 1.0f / (max_value() - min_value()));

    vec4 dummy[3];

    mat4 m_matrix;
    mat4 m_matrix_inverse;
    mat4 m_matrix_inverse_transpose;

    mat4 mv_matrix;
    mat4 mv_matrix_inverse;
    mat4 mv_matrix_inverse_transpose;

    mat4 mvp_matrix;
    mat4 mvp_matrix_inverse;
} volume_data;

// subroutine declaration
subroutine vec4 color_lookup(in vec3 spos);
subroutine uniform color_lookup volume_color_lookup;

subroutine (color_lookup)
vec4 raw_lookup(in vec3 spos)
{
    float v = textureLod(volume_raw, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;

    return vec4((v - volume_data.value_range.x) * volume_data.value_range.w);
}

subroutine (color_lookup)
vec4 raw_color_map_lookup(in vec3 spos)
{
    float v = textureLod(volume_raw, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
    v = (v - volume_data.value_range.x) * volume_data.value_range.w;

    return texture(color_map, vec2(v, 0.0));
    //return vec4(texture(color_map, v).rgb, 1.0);
}

struct ray
{
    vec3 origin;
    vec3 direction;
    vec3 direction_rec;
};

void
make_ray(out ray r,
         in ivec2 spos,
         in  vec2 ssize)
{
    vec4 spos_nrm = vec4((float(spos.x) / ssize.x) * 2.0 - 1.0,
                         (float(spos.y) / ssize.y) * 2.0 - 1.0,
                         -1.0,
                          1.0);

    vec4 spos_os  = volume_data.mvp_matrix_inverse * spos_nrm;
    spos_os.xyz /= spos_os.w;

    r.origin        = volume_data.os_camera_position.xyz;
    r.direction     = normalize(spos_os.xyz - r.origin);
    r.direction_rec = 1.0 / r.direction;
}

bool
ray_box_intersection(in ray    r,
                     in vec3   bbmin,
                     in vec3   bbmax,
                     out float tmin,
                     out float tmax)
{
    float l1 = (bbmin.x - r.origin.x) * r.direction_rec.x;
    float l2 = (bbmax.x - r.origin.x) * r.direction_rec.x;
    tmin = min(l1,l2);
    tmax = max(l1,l2);

    l1   = (bbmin.y - r.origin.y) * r.direction_rec.y;
    l2   = (bbmax.y - r.origin.y) * r.direction_rec.y;
    tmin = max(min(l1,l2), tmin);
    tmax = min(max(l1,l2), tmax);
        
    l1   = (bbmin.z - r.origin.z) * r.direction_rec.z;
    l2   = (bbmax.z - r.origin.z) * r.direction_rec.z;
    tmin = max(min(l1,l2), tmin);
    tmax = min(max(l1,l2), tmax);

    //return ((lmax > 0.f) & (lmax >= lmin));
    //return ((lmax > 0.f) & (lmax > lmin));
    return ((tmin > 0.0) && (tmax > tmin));
}

float
length_sqr(const vec3 a, const vec3 b)
{
    vec3 d = b - a;
    return dot(d, d);
}


// implementation /////////////////////////////////////////////////////////////////////////////////
bool
inside_volume_bounds(const in vec3 sampling_position)
{
    return (   all(greaterThanEqual(sampling_position, vec3(0.0)))
            && all(lessThanEqual(sampling_position, volume_data.volume_extends.xyz)));
}

void main() 
{
#if 1
    ray cur_ray;
    make_ray(cur_ray, ivec2(gl_FragCoord.xy), viewport_size);

    float tmin = 0.0;
    float tmax = 0.0;

    if (ray_box_intersection(cur_ray, vec3(0.0), volume_data.volume_extends.xyz, tmin, tmax)) {

        vec3 cam_pos   = volume_data.os_camera_position.xyz;
        vec3 ray_entry = tmin * cur_ray.direction + cur_ray.origin;
        vec3 ray_exit  = tmax * cur_ray.direction + cur_ray.origin;

        vec3 ray_increment = cur_ray.direction * volume_data.sampling_distance.x;
        vec3 sampling_pos  = ray_entry + ray_increment; // test, increment just to be sure we are in the volume
        vec3 to_tex        = volume_data.scale_obj_to_tex.xyz;

        float smpl_sqr_dist  = length_sqr(cam_pos, sampling_pos);
        float exit_sqr_dist  = length_sqr(cam_pos, ray_exit);

        vec4 dst = vec4(0.0, 0.0, 0.0, 0.0);
        int    loop_count = 0;

#if 1
        while ((exit_sqr_dist - smpl_sqr_dist) > 0.0 && dst.w < 0.99) {
            ++loop_count;
            float  s      = textureLod(volume_raw, sampling_pos * to_tex, 0.0).r;
            vec4   src    = texture(color_map, vec2(s, 0.0));
            //vec4   src = vec4(s, s, s, 0.1);

            // increment ray
            sampling_pos  += ray_increment;
            smpl_sqr_dist  = length_sqr(cam_pos, sampling_pos);

            //inside_volume  = inside_volume_bounds(sampling_pos) && (dst.a < 0.99);

            // opacity correction
            src.w = 1.0 - pow(1.0 - src.w, volume_data.sampling_distance.y);

            // compositing
            float omda_sa = (1.0 - dst.w) * src.w;
            dst.xyz += omda_sa * src.xyz;
            dst.w   += omda_sa;
        }
        out_color = dst;
		//out_color = texture(color_map, vec2(float(loop_count) / 1500.0, 0.0));
#else
        out_color = vec4(ray_exit, 1.0);
#endif
    }
    else {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
    }

#else

    vec3 ray_increment      = normalize(vertex_in.ray_entry_os - volume_data.os_camera_position.xyz) * volume_data.sampling_distance.x;
    vec3 sampling_pos       = vertex_in.ray_entry_os + ray_increment; // test, increment just to be sure we are in the volume

    vec4 dst = vec4(0.0, 0.0, 0.0, 0.0);

    bool inside_volume = inside_volume_bounds(sampling_pos);

#if 1
    while (inside_volume) {
#if 0
        float s  = texture(volume_raw, sampling_pos * volume_data.scale_obj_to_tex.xyz).r;
        vec4 src = texture(color_map, s);
#else
        vec4 src = volume_color_lookup(sampling_pos);
#endif
        // increment ray
        sampling_pos  += ray_increment;
        inside_volume  = inside_volume_bounds(sampling_pos) && (dst.a < 0.99);

        // opacity correction
        src.a = 1.0 - pow(1.0 - src.a, volume_data.sampling_distance.y);

        // compositing
        float omda_sa = (1.0 - dst.a) * src.a;
        dst.rgb += omda_sa*src.rgb;
        dst.a   += omda_sa;
    }
#else
    ray cur_ray;
    make_ray(cur_ray, ivec2(gl_FragCoord.xy), ivec2(1920, 1200));
    dst.xyz = volume_data.mvp_matrix_inverse[0].xyz;//cur_ray.direction;//
    dst.a   = 1.0;
#endif
    out_color = dst;
#endif
}
