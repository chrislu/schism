
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 420 core

#extension GL_ARB_shading_language_include : require

#if SCM_TEXT_NV_BINDLESS_TEXTURES == 1
#extension GL_NV_bindless_texture : require
#extension GL_NV_gpu_shader5      : require
#endif

#define SCM_TEST_NV_BINDLESS_TEX_BUFFER     1
#define SCM_TEST_NV_BINDLESS_TEX_BUFFER_PRE 0

#include </scm/gl_util/camera_block.glslh>

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
#if SCM_TEXT_NV_BINDLESS_TEXTURES != 1
uniform sampler3D volume_raw;
uniform sampler1D color_map;
#else

layout (binding = 4) uniform usamplerBuffer texture_handles;

#if SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1 && SCM_TEST_NV_BINDLESS_TEX_BUFFER_PRE == 1
uint64_t vtex_smpl = 0;
uint64_t ctex_smpl = 0;
#endif // SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1

#endif // SCM_TEXT_NV_BINDLESS_TEXTURES != 1

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

#if SCM_TEXT_NV_BINDLESS_TEXTURES == 1
    sampler3D volume_texture;
    sampler1D color_map;
#endif // SCM_TEXT_NV_BINDLESS_TEXTURES == 1
} volume_data;

// subroutine declaration
subroutine vec4 color_lookup(in vec3 spos);
subroutine uniform color_lookup volume_color_lookup;

subroutine (color_lookup)
vec4 raw_lookup(in vec3 spos)
{
#if SCM_TEXT_NV_BINDLESS_TEXTURES == 1
    float v = textureLod(volume_data.volume_texture, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
#else // SCM_TEXT_NV_BINDLESS_TEXTURES == 1
    float v = textureLod(volume_raw, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
#endif // SCM_TEXT_NV_BINDLESS_TEXTURES == 1
    //v *= 65535.0/4096.0;

    return vec4((v - volume_data.value_range.x) * volume_data.value_range.w);
}

subroutine (color_lookup)
vec4 raw_color_map_lookup(in vec3 spos)
{
#if SCM_TEXT_NV_BINDLESS_TEXTURES == 1
#if SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1
#if SCM_TEST_NV_BINDLESS_TEX_BUFFER_PRE == 1
    float v = textureLod(sampler3D(vtex_smpl), spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
    v = (v - volume_data.value_range.x) * volume_data.value_range.w;

    return texture(sampler1D(ctex_smpl), v);
#else // SCM_TEST_NV_BINDLESS_TEX_BUFFER_PRE == 1
    uvec2     vtex_hndl_enc = texelFetch(texture_handles, 0).xy;
    uint64_t  vtex_hndl     = packUint2x32(vtex_hndl_enc);
    sampler3D vtex_smpl     = sampler3D(vtex_hndl);
    float v = textureLod(vtex_smpl, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
    v = (v - volume_data.value_range.x) * volume_data.value_range.w;

    uvec2     ctex_hndl_enc = texelFetch(texture_handles, 1).xy;
    uint64_t  ctex_hndl     = packUint2x32(ctex_hndl_enc);
    sampler1D ctex_smpl     = sampler1D(ctex_hndl);

    return texture(ctex_smpl, v);
#endif // SCM_TEST_NV_BINDLESS_TEX_BUFFER_PRE == 1
#else // SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1
    float v = textureLod(volume_data.volume_texture, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
    v = (v - volume_data.value_range.x) * volume_data.value_range.w;

    return texture(volume_data.color_map, v);
#endif // SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1
#else // SCM_TEXT_NV_BINDLESS_TEXTURES == 1
    float v = textureLod(volume_raw, spos * volume_data.scale_obj_to_tex.xyz, volume_lod).r;
    v = (v - volume_data.value_range.x) * volume_data.value_range.w;

    return texture(color_map, v);
#endif // SCM_TEXT_NV_BINDLESS_TEXTURES == 1
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
#if SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1 && SCM_TEST_NV_BINDLESS_TEX_BUFFER_PRE == 1
    uvec2     vtex_hndl_enc = texelFetch(texture_handles, 0).xy;
    uvec2     ctex_hndl_enc = texelFetch(texture_handles, 1).xy;
    vtex_smpl     = packUint2x32(vtex_hndl_enc);
    ctex_smpl     = packUint2x32(ctex_hndl_enc);
#endif // SCM_TEST_NV_BINDLESS_TEX_BUFFER == 1


    vec3 ray_increment      = normalize(vertex_in.ray_entry_os - volume_data.os_camera_position.xyz) * volume_data.sampling_distance.x;
    vec3 sampling_pos       = vertex_in.ray_entry_os + ray_increment; // test, increment just to be sure we are in the volume

    vec4 dst = vec4(0.0, 0.0, 0.0, 0.0);

    bool inside_volume = inside_volume_bounds(sampling_pos);

    while (inside_volume) {
        vec4 src = volume_color_lookup(sampling_pos);

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
