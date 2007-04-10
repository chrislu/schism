
#extension ARB_texture_rectangle : require
#extension ARB_texture_rectangle : enable

#version 120

uniform sampler2DRect   _geometry_image;
uniform sampler2DRect   _correction_image;

void main(void)
{
    vec2 offset = texture2DRect(_correction_image,  vec2(gl_FragCoord.xy)).rg;
    
    vec4 offset_color = texture2DRect(_geometry_image, ((vec2(gl_FragCoord.xy) + offset * vec2(255.0))));

    gl_FragColor = /*vec4(length(offset), 0.0, 0.0, 0.0) + */offset_color;//
}


