
#version 120

varying vec3 normal;

uniform sampler2D _diff_gloss;
uniform sampler2D _normal;

void main() 
{
    vec4 diff_gloss = texture2D(_diff_gloss, gl_TexCoord[0].xy);

    // unpack normal from texture color
    vec3 normal     = texture2D(_normal,     gl_TexCoord[0].xy).xyz * 2.0 - vec3(1.0);
    normal = normalize(gl_NormalMatrix * normal);

     
    gl_FragData[0] = diff_gloss;
    gl_FragData[1] = vec4(normal, 0.0);
}
