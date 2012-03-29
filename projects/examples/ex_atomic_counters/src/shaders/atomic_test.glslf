
#version 420 core

// input layout definitions ///////////////////////////////////////////////////////////////////////

// input/output definitions ///////////////////////////////////////////////////////////////////////
in per_vertex {
    vec3 normal;
    vec2 texcoord;
    vec3 view_dir;
} v_in;

// attribute layout definitions ///////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;

// uniform input definitions //////////////////////////////////////////////////////////////////////
layout(binding = 0, offset = 0) uniform atomic_uint frag_count;

uniform vec2 screen_res;
// global constants ///////////////////////////////////////////////////////////////////////////////
const float epsilon                 = 0.0001;

const vec3 white = vec3(1.0, 1.0, 1.0);
const vec3 black = vec3(0.0, 0.0, 0.0);
const vec3 red   = vec3(1.0, 0.0, 0.0);
const vec3 green = vec3(0.0, 1.0, 0.0);
const vec3 blue  = vec3(0.0, 0.0, 1.0);
const vec3 lblue = vec3(0.2, 0.7, 0.9);

// implementation /////////////////////////////////////////////////////////////////////////////////
void main() 
{
#if 1
	uint m = (1u << 8u) - 1u;
    uint c = atomicCounterIncrement(frag_count) + 1;

	out_color.rgb = vec3(float((c & (m <<  0u)) % 255u) / 255.0,
		                 float((c & (m <<  8u)) % 255u) / 255.0,
		                 float((c & (m << 16u)) % 255u) / 5.0);

    //out_color.rgb = vec3(float(c) / (screen_res.x * screen_res.y * 2.0));
    out_color.a   = 1.0;

#else
    vec4 c = vec4(lblue, 1.0);//texture(tex_color, v_in.texcoord);

    vec3 n = normalize(v_in.normal); 
    vec3 l = normalize(vec3(1.0, 1.0, 1.0)); // assume parallel light!
    vec3 v = normalize(v_in.view_dir);
    vec3 h = normalize(l + v);

    out_color.rgb =    c.rgb * (dot(n, l) * 0.5 + 0.5)//max(0.0, dot(n, l))
                     + lblue * pow(max(0.0, dot(n, h)), 120.0);
    
    //out_color.rgb =  + c.rgb;
    out_color.a   = 1.0;
#endif
}

