
#version 420 core

#if __VERSION__ >= 420
#else
#extension GL_EXT_shader_image_load_store : require
#extension GL_ARB_gpu_shader5 : require
#endif

// input layout definitions ///////////////////////////////////////////////////////////////////////
layout(early_fragment_tests) in;

// input/output definitions ///////////////////////////////////////////////////////////////////////
in per_vertex {
    vec3 normal;
    vec2 texcoord;
} v_in;

// attribute layout definitions ///////////////////////////////////////////////////////////////////
layout(location = 0, index = 0) out vec4 out_color;

// uniform input definitions //////////////////////////////////////////////////////////////////////
#ifdef SCM_TEST_TEXTURE_IMAGE_STORE

#if __VERSION__ >= 420
layout(binding = 0, rgba8) writeonly uniform image2D       output_image;
layout(binding = 1, r32ui) coherent  uniform uimage2D      depth_image;
layout(binding = 2, r32ui) coherent  uniform uimage2D      lock_image;
layout(binding = 3, rgba8) writeonly uniform imageBuffer   output_buffer;
#else
layout(size2x32) coherent uniform uimage2D      output_image;
layout(size1x32) coherent uniform uimage2D      depth_image;
layout(size1x32) coherent uniform uimage2D      lock_image;
layout(size1x32) coherent uniform imageBuffer   output_buffer;
#endif

uniform ivec2   output_res;

bool lock(in ivec2 frag_coord)   { return imageAtomicExchange(lock_image, frag_coord, 1u) == 0u; }
void unlock(in ivec2 frag_coord) {        imageAtomicExchange(lock_image, frag_coord, 0u); }

#endif // SCM_TEST_TEXTURE_IMAGE_STORE

// implementation /////////////////////////////////////////////////////////////////////////////////
void main() 
{
    vec4 c = vec4(v_in.normal, 1.0);

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE
#if 0
    ivec2 frag_coord = ivec2(gl_FragCoord.xy);
    if (all(equal(frag_coord % SCM_FEEDBACK_BUFFER_REDUCTION, ivec2(0)))) {
        ivec2 out_coord = frag_coord / SCM_FEEDBACK_BUFFER_REDUCTION;
        bool done = false;
        while (!done) {
            if (lock(out_coord)) {
                float z     = gl_FragCoord.z;
                float cur_z = imageLoad(depth_image, out_coord).x;
                if (z > 0.0 && z < cur_z) {
                    imageStore(depth_image, out_coord, vec4(z, 0.0, 0.0, 0.0));
                    imageStore(output_image, out_coord, c);
                }
                done = true;
                unlock(out_coord);
            }
        }
    }
    else {
        discard;
    }
#elif 1
    ivec2 frag_coord = ivec2(gl_FragCoord.xy);
    if (all(equal(frag_coord % SCM_FEEDBACK_BUFFER_REDUCTION, ivec2(0)))) {
        ivec2 out_coord = frag_coord / SCM_FEEDBACK_BUFFER_REDUCTION;
        uint z          = uint((1.0 - clamp(gl_FragCoord.z, 0.0, 1.0)) * float(0x00EFFFFFu));
        if (z > imageAtomicMax(depth_image, out_coord, z)) {
            //imageStore(depth_image, out_coord, vec4(z, 0.0, 0.0, 0.0));
            imageStore(output_image, out_coord, c);
            //memoryBarrier();
        }
    }
    else {
        discard;
    }
#elif 0
    ivec2 frag_coord = ivec2(gl_FragCoord.xy);
    if (all(equal(frag_coord % SCM_FEEDBACK_BUFFER_REDUCTION, ivec2(0)))) {
        ivec2 out_coord = frag_coord / SCM_FEEDBACK_BUFFER_REDUCTION;
        uint z          = uint((1.0 - clamp(gl_FragCoord.z, 0.0, 1.0)) * float(0x00EFFFFFu));
        if (z > imageAtomicMax(depth_image, out_coord, z)) {
            //imageStore(output_image, out_coord, c);
            imageStore(output_buffer, out_coord.y * (1600 / SCM_FEEDBACK_BUFFER_REDUCTION) /*output_res.x*/ + out_coord.x, c);
            //memoryBarrier();
        }
    }
    else {
        discard;
    }
#elif 0
    ivec2 frag_coord = ivec2(gl_FragCoord.xy);
    ivec2 out_coord = frag_coord / SCM_FEEDBACK_BUFFER_REDUCTION;
    //imageStore(output_image, out_coord, c);
    imageStore(output_buffer, out_coord.y * (1600 / SCM_FEEDBACK_BUFFER_REDUCTION) /*output_res.x*/ + out_coord.x, c);
    //memoryBarrier();
#else
    ivec2 frag_coord = ivec2(gl_FragCoord.xy);
    ivec2 out_coord = frag_coord;
    imageStore(output_image, out_coord, c);
    //memoryBarrier();
#endif
#endif // SCM_TEST_TEXTURE_IMAGE_STORE
    out_color = c;
}

