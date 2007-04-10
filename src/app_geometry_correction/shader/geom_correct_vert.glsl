
#version 120

void main()
{
    gl_Position     = ftransform();//gl_ModelViewProjectionMatrix * gl_Vertex;
}
