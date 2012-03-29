
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

// eye space variables
varying vec3 eyeSpaceTangent;
varying vec3 eyeSpaceBinormal;
varying vec3 eyeSpaceNormal;

varying vec3 eyeSpaceVert;
varying vec3 eyeSpaceLight;

void main(void)
{
    // transform directions into eye space
	eyeSpaceTangent  = gl_NormalMatrix * vec3(1.0, 0.0, 0.0);
	eyeSpaceBinormal = gl_NormalMatrix * vec3(0.0, 1.0, 0.0);
   	eyeSpaceNormal   = gl_NormalMatrix * vec3(0.0, 0.0, 1.0);

   	// transform eye point into eye space
   	eyeSpaceVert = (gl_ModelViewMatrix * gl_Vertex).xyz;

   	//// transform light point into eye space
   	eyeSpaceLight = (gl_ModelViewMatrix * vec4(gl_LightSource[0].position.xyz,1.0)).xyz;

   	// get texture coordinates and pixel position for the fragment
   	gl_TexCoord[0] = gl_MultiTexCoord0;
   	gl_TexCoord[0].z = -gl_Vertex.z;

   	gl_Position = ftransform();
}
