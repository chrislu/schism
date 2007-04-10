
#version 120

varying vec3 _cam_pos;
varying vec3 _ray_dir;
varying vec3 _obj_pos;

void main()
{
    vec4 cam_pos = gl_TextureMatrix[0] * gl_ModelViewMatrixInverse[3];
    vec4 obj_pos = gl_TextureMatrix[0] * gl_MultiTexCoord0;

    _cam_pos    = cam_pos.xyz / cam_pos.w;
    _obj_pos    = obj_pos.xyz / obj_pos.w;

    _ray_dir    = _obj_pos - _cam_pos;

    gl_Position = ftransform();
}