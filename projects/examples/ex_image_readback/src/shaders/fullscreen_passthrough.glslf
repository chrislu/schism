
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

uniform sampler2DRect   _image;

void main()
{
    gl_FragColor = texture(_image, gl_FragCoord.xy);
}
