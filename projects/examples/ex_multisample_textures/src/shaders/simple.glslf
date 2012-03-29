
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#version 420 core

layout(location = 0, index = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D env_map;
layout(binding = 1) uniform sampler2D diff_map;
layout(binding = 2) uniform sampler2D normal_map;
layout(binding = 3) uniform sampler2D diffuse_map;

uniform mat4 projection_matrix;
uniform mat4 model_view_matrix;

in per_vertex {
    smooth vec3 es_position;
    smooth vec3 es_normal;
    smooth vec3 os_normal;
    smooth vec3 os_tangent;
    smooth vec3 os_binormal;
    smooth vec2 texcoord;
} v_in;

#define USE_NORMAL_MAPPING 1
//#undef  USE_NORMAL_MAPPING

#define USE_OBJ_MATERIAL_DEF 1
#undef  USE_OBJ_MATERIAL_DEF

#if USE_OBJ_MATERIAL_DEF
//uniform vec3    light_ambient;
//uniform vec3    light_diffuse;
//uniform vec3    light_specular;
//uniform vec3    light_position;

uniform vec3    material_ambient;
uniform vec3    material_diffuse;
uniform vec3    material_specular;
uniform float   material_shininess;
uniform float   material_opacity;
#endif // USE_OBJ_MATERIAL_DEF

const float pi    = 3.1415926535897932384626433832795;
const float invpi = 0.31830988618379067153776752674503;
const float fresnel_coeff  = 0.1;
const vec3  material_color = vec3(0.5);//0.3, 0.1, 0.1);//vec3(0.2, 0.2, 0.2);

float
fresnel(in vec3 v, in vec3 n)
{
#if USE_OBJ_MATERIAL_DEF
    const float f = material_shininess / 255.0; // fresnel_coeff
#else
    const float f = fresnel_coeff;
#endif
    float ret = (1.0 - dot(v, n));
    ret = f + (1.0 - f) * pow(ret, 5.0);
    return ret;
}

vec2
env_long_lat(in vec3 v)
{
    vec2 a_xz = normalize(v.xz);
    vec2 a_yz = normalize(v.yz);

    return vec2(0.5 * (1.0 + invpi * atan(a_xz.x, -a_xz.y)),
                acos(-v.y) * invpi);
}

void main()
{
#if USE_NORMAL_MAPPING
    vec4 mat_col  = texture(diffuse_map, v_in.texcoord);
    vec3 ts_nrml  = (texture(normal_map, v_in.texcoord).xyz * 2.0) - 1.0;
    mat3 os_to_ts = mat3(v_in.os_tangent, v_in.os_binormal, v_in.os_normal);
    vec3 os_nrml  = inverse(transpose(os_to_ts)) * ts_nrml;
    vec3 nrml = normalize((transpose(inverse(model_view_matrix)) * vec4(os_nrml, 0.0)).xyz);//normalize(v_in.es_normal);//
#else
#if USE_OBJ_MATERIAL_DEF
    vec4 mat_col = vec4(material_diffuse, 1.0);
    vec3 nrml = normalize(v_in.es_normal);//
#else
    vec4 mat_col = vec4(material_color, 1.0);
    vec3 nrml = normalize(v_in.es_normal);//
#endif
#endif

    vec3 vvec = normalize(v_in.es_position);
    vec3 refl = reflect(vvec, nrml);

    vec4 env_col    = vec4(texture(env_map, env_long_lat(refl)).rgb, 1.0);
    vec4 diff_coeff = vec4(texture(diff_map, env_long_lat(nrml)).rgb, 1.0);

#if USE_OBJ_MATERIAL_DEF
    out_color.rgb = mix(mat_col * diff_coeff, env_col, vec4(material_specular, 1.0) * fresnel(-vvec, nrml)).rgb;
    out_color.a   = material_opacity;
#else
    out_color = mix(mat_col * diff_coeff, env_col, fresnel(-vvec, nrml));
#endif


}

