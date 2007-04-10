
#version 110

varying vec3 normal;
varying vec3 view_dir;

void main()
{
    normal          =  normalize(gl_NormalMatrix * gl_Normal);
    view_dir        = -normalize(gl_ModelViewMatrix * gl_Vertex).xyz;
    // pass texture coordinates along to fragment program
    gl_TexCoord[0]  = gl_MultiTexCoord0;

    gl_Position     = gl_ModelViewProjectionMatrix * gl_Vertex;
 }
