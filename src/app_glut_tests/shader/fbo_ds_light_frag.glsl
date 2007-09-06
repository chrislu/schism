
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
uniform sampler2DRect   _normal;

//varyings


// implementation
vec3 backproject_depth(const in float depth)
{
    vec4 clip_space_pos = vec4(vec2(2.0) * gl_FragCoord.xy / _viewport_dim - vec2(1.0),
                               2.0 * depth - 1.0,
                               1.0);

    vec4 eye_space_pos  = _projection_inverse * clip_space_pos;

    return (eye_space_pos.xyz / eye_space_pos.w);
}


void main()
{
    // fetch data from gbuffers
    vec3 position       = backproject_depth(texture2DRect(_depth, gl_FragCoord.xy).x);
    vec3 view_dir       = -normalize(position);
    vec3 normal         = texture2DRect(_normal, gl_FragCoord.xy).xyz;
    vec4 diff_gloss     = texture2DRect(_color_gloss, gl_FragCoord.xy);

    // lighting calculation
    vec3 l = normalize(gl_LightSource[0].position.xyz); // parallel light assumed 
    vec3 h = normalize(l + view_dir);

    vec3 res =  diff_gloss.xyz * gl_LightSource[0].diffuse.xyz * max(0.0, dot(normal, l))
              + diff_gloss.a * gl_LightSource[0].specular.xyz * pow(max(0.0, dot(normal, h)), 128.0);


    //output
    gl_FragColor = vec4(res, 1.0);
}
