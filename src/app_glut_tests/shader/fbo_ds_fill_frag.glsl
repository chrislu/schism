
#version 120

varying vec3 normal;

uniform sampler2D _diff_gloss;
uniform sampler2D _normal;

void main() 
{
    // fetch color and gloss
    vec4 diff_gloss = vec4(0.2, 0.5, 1.0, 1.0);//texture2D(_diff_gloss, gl_TexCoord[0].xy);

    // fetch normal
    // unpack normal from texture color
    //vec3 normal     = texture2D(_normal,     gl_TexCoord[0].xy).xyz * 2.0 - vec3(1.0);
    //normal = normalize(gl_NormalMatrix * normal);

    // outpt data into the gbuffers
    gl_FragData[0] = diff_gloss;

    gl_FragData[1] = vec4(gl_FrontFacing ? normalize(normal) : -normalize(normal), 0.0);
}
