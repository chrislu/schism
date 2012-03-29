
#version 420 core

#extension GL_ARB_shading_language_include : require

#include </scm/gl_util/camera_block.glsl>

// output layout definitions //////////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;

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
uniform sampler1D color_map;

uniform float volume_lod;

layout(std140, column_major) uniform;

uniform volume_uniform_data
{
    vec4 volume_extends;     // w unused
    vec4 scale_obj_to_tex;   // w unused
    vec4 sampling_distance;  // x - os sampling distance, y opacity correction factor, zw unused
    vec4 os_camera_position;
    vec4 value_range;        // vec4f(min_value(), max_value(), max_value() - min_value(), 1.0f / (max_value() - min_value()));

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

    //v *= 65535.0/4096.0;

    return vec4((v - volume_data.value_range.x) * volume_data.value_range.w);
}

subroutine (color_lookup)
vec4 raw_color_map_lookup(in vec3 spos)
{
    float v = textureLod(volume_raw, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
    v = (v - volume_data.value_range.x) * volume_data.value_range.w;

    //v *= 65535.0/4096.0;

    return texture(color_map, v);
    //return vec4(texture(color_map, v).rgb, 1.0);
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
    vec3 ray_increment      = normalize(vertex_in.ray_entry_os - volume_data.os_camera_position.xyz) * volume_data.sampling_distance.x;
    vec3 sampling_pos       = vertex_in.ray_entry_os + ray_increment; // test, increment just to be sure we are in the volume

    vec4 dst = vec4(0.0, 0.0, 0.0, 0.0);

    bool inside_volume = inside_volume_bounds(sampling_pos);

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

    out_color = dst;
}
