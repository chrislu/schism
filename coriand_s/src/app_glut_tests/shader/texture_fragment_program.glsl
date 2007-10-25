
#version 110

varying vec3 normal;
varying vec3 view_dir;

uniform sampler2D _diff_gloss;
uniform sampler2D _normal;
uniform float     _shininess;

#define USE_NORMAL_MAPPING 1

void main() 
{
    vec4 diff_gloss = texture2D(_diff_gloss, gl_TexCoord[0].xy);

#if USE_NORMAL_MAPPING == 1
    // unpack normal from texture color
    vec3 normal     = texture2D(_normal,     gl_TexCoord[0].xy).xyz * 2.0 - vec3(1.0);
    normal = normalize(gl_NormalMatrix * normal);

    vec3 l = normalize(gl_LightSource[0].position.xyz); // parallel light assumed 
    vec3 v = normalize(view_dir);
    vec3 h = normalize(l + v);

    vec3 res =  diff_gloss.xyz * gl_LightSource[0].diffuse.xyz * max(0.0, dot(normal, l))
              + diff_gloss.a * gl_LightSource[0].specular.xyz * pow(max(0.0, dot(normal, h)), _shininess);
         
    gl_FragColor = vec4(res, 1.0);

#else
    vec3 n = normalize(normal); 
    vec3 l = normalize(gl_LightSource[0].position.xyz); // parallel light assumed 
    vec3 v = normalize(view_dir);

    vec3 h = normalize(l + v);

    vec3 res =  diff_gloss.xyz * gl_LightSource[0].diffuse.xyz * max(0.0, dot(n, l))
              + diff_gloss.a * gl_LightSource[0].specular.xyz * pow(max(0.0, dot(n, h)), _shininess);

    gl_FragColor = vec4(res.xyz, 1.0);
#endif
}

