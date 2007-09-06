
#version 120

varying vec3 normal;

void main()
{
    normal          =  normalize(gl_NormalMatrix * gl_Normal);
    // pass texture coordinates along to fragment program
    gl_TexCoord[0]  = gl_MultiTexCoord0;

    gl_Position     = gl_ModelViewProjectionMatrix * gl_Vertex;
 }
