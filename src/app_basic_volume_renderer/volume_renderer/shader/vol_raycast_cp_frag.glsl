
#version 110

uniform sampler3D   _volume;
uniform sampler1D   _color_alpha;

vec4 do_shading(in vec4 inp)
{
    return (texture1D(_color_alpha, inp.r));
}

void main()
{
    gl_FragColor = vec4(do_shading(texture3D(_volume, gl_TexCoord[0].xyz)).rgb, 1.0);
}
