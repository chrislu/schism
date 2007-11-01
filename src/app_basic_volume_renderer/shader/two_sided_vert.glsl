
#version 120

varying vec3 normal;
varying vec3 view_dir;

varying vec3 unc_color;

uniform sampler3D   _unc_texture;

uniform mat4        _vert2unit;
uniform mat4        _vert2vol;
uniform mat4        _vert2vol_it;

uniform float       _anim_step;

const vec3 red      = vec3(1.0, 0.0, 0.0);
const vec3 green    = vec3(0.0, 1.0, 0.0);


void main()
{
    normal    =  normalize(mat3(_vert2vol_it) * gl_Normal);//gl_NormalMatrix
    view_dir  = -normalize(gl_ModelViewMatrix * _vert2vol * gl_Vertex).xyz;

    //unc_color = mix(green, red, 1.0 - texture3D(_unc_texture, (_vert2unit * gl_Vertex).xyz).r);//gl_Color.rgb;//
    unc_color = gl_Color.rgb;//mix(green, red, _anim_step * (1.0 - clamp(pow(texture3D(_unc_texture, (_vert2unit * gl_Vertex).xyz).r, 2.0), 0.0, 1.0)));
    //
    gl_Position = gl_ModelViewProjectionMatrix * _vert2vol * gl_Vertex;
}

