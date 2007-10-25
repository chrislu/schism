
#version 110

uniform vec2      _win_dim;
uniform sampler2D _image;

const vec2 frag_off = vec2(1.0) / _win_dim;

#define KERNEL_SIZE 9

// Gaussian kernel
// 1 2 1
// 2 4 2
// 1 2 1	
float kernel[KERNEL_SIZE] = { 1.0/16.0, 2.0/16.0, 1.0/16.0,
				              2.0/16.0, 4.0/16.0, 2.0/16.0,
				              1.0/16.0, 2.0/16.0, 1.0/16.0 };


vec2 offset[KERNEL_SIZE] = { vec2(-frag_off.x, -frag_off.y), vec2(0.0, -frag_off.y), vec2(frag_off.x, -frag_off.y), 
				             vec2(-frag_off.x, 0.0),         vec2(0.0, 0.0),         vec2(frag_off.x, 0.0), 
				             vec2(-frag_off.x, frag_off.y),  vec2(0.0, frag_off.y),  vec2(frag_off.x, frag_off.y) };
void main(void)
{
    int i = 0;
    vec4 image = vec4(0.0);
   
    for(i=0; i < KERNEL_SIZE; ++i)
    {
        image += texture2D(_image, gl_TexCoord[0].xy + offset[i]) * kernel[i];
    }

    gl_FragColor = image;
}


