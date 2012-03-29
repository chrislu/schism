
uniform sampler2DRect   _image;

void main()
{
    gl_FragColor = texture(_image, gl_FragCoord.xy);
}
