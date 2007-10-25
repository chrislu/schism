
#version 120

varying vec3 normal;
varying vec3 view_dir;

void main()
{
    normal    =  normalize(gl_NormalMatrix * gl_Normal);
    view_dir  = -normalize(gl_ModelViewMatrix * gl_Vertex).xyz;

    gl_Position = ftransform();
}

