
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * gl_Vertex;
}
