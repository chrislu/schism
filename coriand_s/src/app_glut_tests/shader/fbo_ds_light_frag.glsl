
#version 120

#pragma optimize(on)
#pragma debug(off)

#extension ARB_texture_rectangle : require
#extension ARB_texture_rectangle : enable

// uniforms
uniform vec2            _viewport_dim;

uniform mat4            _projection_inverse;

uniform sampler2DRect   _depth;
uniform sampler2DRect   _color_gloss;
uniform sampler2DRect   _specular_shininess;
uniform sampler2DRect   _normal;

//varyings

/*
 * defines
 */

#define MAXIMUM_PERFORMANCE
#undef MAXIMUM_PERFORMANCE

#define SUPPORT_ALL_LIGHTS

#if defined(MAXIMUM_PERFORMANCE)
#  undef SUPPORT_ALL_LIGHTS
#endif

/*
 * constants
 */

const vec4 const_color_black = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 const_color_white = vec4(1.0, 1.0, 1.0, 1.0);

#if defined(SUPPORT_ALL_LIGHTS)
const int const_max_lights = gl_MaxLights;
#else
const int const_max_lights = (gl_MaxLights / 2);
#endif

/*
 * types
 */

struct gl_LightConeParameters {
  float cos_inner_angle;
  float cos_outer_angle;
  float diff_inner_outer;
  bool  compute_falloff_spot;
};

/*
 * uniform variables
 */

uniform bool      light_state[gl_MaxLights];
uniform sampler2D base_texture;

/*
 * varying variables
 */

// implementation
vec3 backproject_depth(const in float depth)
{
    vec4 clip_space_pos = vec4(gl_FragCoord.xy / _viewport_dim, depth, 1.0) * 2.0 - 1.0;
//vec4 clip_space_pos = vec4(vec2(2.0) * gl_FragCoord.xy / _viewport_dim - vec2(1.0),
//	                               2.0 * depth - 1.0,
//	                               1.0);
    vec4 eye_space_pos  = _projection_inverse * clip_space_pos;

    return (eye_space_pos.xyz / eye_space_pos.w);
}

/*
 * Cg compatibility
 */

vec4
lit(const in float NdotL, const in float NdotH, const in float exponent)
{
  return vec4(1.0,                                                    // .x: ambient contrib
              NdotL * 0.5 + 0.5, //max(NdotL, 0.0),                   // .y: diffuse contrib
              pow(NdotH * 0.5 +0.5, exponent),//(min(NdotL, NdotH) < 0.0) ? 0.0 : pow(NdotH, exponent), // .z: specular contrib
              1.0);                                                   // .w: unused
}

float
saturate(in float x)
{
  return clamp(x, 0.0, 1.0);
}

/*
 * shared functions
 */

vec3
get_light_vector(const in int lindex, const in vec3 object_pos)
{
  return ((0.0 != gl_LightSource[lindex].position.w)
          ? (gl_LightSource[lindex].position.xyz - object_pos) // directional: lvec = lpos - opos
          : gl_LightSource[lindex].position.xyz);              // omni:        lvec = lpos
}

float
get_light_attenuation(const in int lindex, const in float dist)
{
  const float attenuation_at_lpos_infty = 1.0;

#if !defined(MAXIMUM_PERFORMANCE)
  return ((0.0 != gl_LightSource[lindex].position.w)
          ? 1.0 / (gl_LightSource[lindex].constantAttenuation +
                   (gl_LightSource[lindex].linearAttenuation * dist) +
                   (gl_LightSource[lindex].quadraticAttenuation * dist * dist))
          : attenuation_at_lpos_infty);
#else
  return attenuation_at_lpos_infty;
#endif
}

gl_LightConeParameters
setup_light_cone(const in int lindex)
{
  gl_LightConeParameters result;
  
  result.compute_falloff_spot = ((0.0 != gl_LightSource[lindex].position.w) &&
                                 (-1.0 != gl_LightSource[lindex].spotCosCutoff));
  
  if (result.compute_falloff_spot) {
    result.cos_outer_angle = saturate(gl_LightSource[lindex].spotCosCutoff);
    // inner = clamp((0.3 * outer), 0.0, 1.0); factor 0.3 is hard-wired for now
    result.cos_inner_angle = saturate(cos(radians((1.0 - gl_LightSource[lindex].spotExponent/128.0) * gl_LightSource[lindex].spotCutoff)));
    result.diff_inner_outer = (result.cos_inner_angle - result.cos_outer_angle);
  }
  
  return result;
}

void
main()
{
    // chrislu
    vec4 dbg_col;
    vec3 position       = backproject_depth(texture2DRect(_depth, gl_FragCoord.xy).r);
    //vec3 view_dir       = -normalize(position);
    vec3 normal         = texture2DRect(_normal, gl_FragCoord.xy).xyz;
    vec4 diff_gloss     = texture2DRect(_color_gloss, gl_FragCoord.xy);
    vec4 spec_shin      = texture2DRect(_specular_shininess, gl_FragCoord.xy);

    if (diff_gloss.a == 0.0) {
        discard;
    }


    const vec3 N = normal; // normalize(normal); the normal is normalized for this fragment
  
    vec4 ambient  = gl_LightModel.ambient;
    vec4 diffuse  = const_color_black;
    vec4 specular = const_color_black;
  
    for (int lindex = 0; lindex < const_max_lights; ++lindex) {
        if (light_state[lindex]) {
            vec3  L           = get_light_vector(lindex, position);
            float attenuation = 1.0;//get_light_attenuation(lindex, length(L));

            L = normalize(L);
            //dbg_col.xyz = position;//(L + 1.0) * 0.5;

            const float NdotL = dot(N, L);//max(0.0, dot(N, L));

            /*if (0.0 < NdotL) */{
#if !defined(MAXIMUM_PERFORMANCE)
                const gl_LightConeParameters light_cone = setup_light_cone(lindex);
        
                if (light_cone.compute_falloff_spot) {
                    const float cos_cur_angle = dot(-L, normalize(gl_LightSource[lindex].spotDirection));
          
                    // avoids dynamic branching
                    attenuation *= saturate((cos_cur_angle - light_cone.cos_outer_angle) /
                                            light_cone.diff_inner_outer);
                }
#endif
        
            const vec3  H = normalize(L - normalize(position.xyz));
            const float NdotH = dot(N, H);//max(0.0, dot(N, H));
            const vec4  lit_result = lit(NdotL, NdotH, spec_shin.w * 66.0); // gl_LightSource[lindex].spotExponent);

            ambient += (attenuation *
                        gl_LightSource[lindex].ambient);

            diffuse += (attenuation *
                        gl_LightSource[lindex].diffuse *
                        diff_gloss * //gl_FrontMaterial.diffuse *
                        lit_result.y);

            specular += (attenuation *
                         gl_LightSource[lindex].specular *
                         spec_shin * //gl_FrontMaterial.specular *
                         lit_result.z);
            }
        }
    }
  
    //ambient *= gl_FrontMaterial.ambient;

    //const vec4 emission = gl_FrontMaterial.emission;
#if 1
    gl_FragColor.rgb = (/*emission + */ambient + diffuse + specular).rgb;
#else
    gl_FragColor.rgb = dbg_col.xyz;
#endif
    gl_FragColor.a   = 1.0; // gl_FrontMaterial.diffuse.a;
  
}



#if 0
void main()
{
    // fetch data from gbuffers
    vec3 position       = backproject_depth(texture2DRect(_depth, gl_FragCoord.xy).x);
    vec3 view_dir       = -normalize(position);
    vec3 normal         = texture2DRect(_normal, gl_FragCoord.xy).xyz;
    vec4 diff_gloss     = texture2DRect(_color_gloss, gl_FragCoord.xy);
    vec4 spec_shin      = texture2DRect(_specular_shininess, gl_FragCoord.xy);

    // lighting calculation
    vec3 l = normalize(gl_LightSource[0].position.xyz); // parallel light assumed 
    vec3 h = normalize(l + view_dir);

    vec3 res =  diff_gloss.xyz * gl_LightSource[0].diffuse.xyz * max(0.0, dot(normal, l))
              + diff_gloss.a * spec_shin.rgb * gl_LightSource[0].specular.xyz * pow(max(0.0, dot(normal, h)), spec_shin.w);


    //output
    gl_FragColor = vec4(res, 1.0);
}

#endif
