
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_ray_cast.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cutil/cutil_math.h>

#include <renderer/volume_uniform_data.h>

#define SCM_LDATA_CUDA_VIS_PROFILE_CLOCK 0 
#define SCM_LDATA_CUDA_VIS_ITER_COUNT    0
#define SCM_LDATA_CUDA_VIS_DEBUG         0

#define SCM_LDATA_CUDA_VIS_SS_COUNT      4 // supported modes: 1, 4, 8

namespace scm {
namespace cuda {

__device__ __constant__ float2 ss_pixel_offsets_01  = {0.5f, 0.5f};
__device__ __constant__ float  ss_sample_offsets_01 = 0.00f;

    // regular grid
    //const float2 ss_pixel_offsets[4] = {{0.25f, 0.25f},
    //                                    {0.75f, 0.25f},
    //                                    {0.25f, 0.75f},
    //                                    {0.75f, 0.75f}};
    // rotated grid grid
#define ss_grid_res 0.125f
//__device__ __constant__ float2 ss_pixel_offsets[4] = {{0.5f, 0.5f},
//                                                      {0.5f, 0.5f},
//                                                      {0.5f, 0.5f},
//                                                      {0.5f, 0.5f}};
//__device__ __constant__ float ss_sample_offsets[4] = {0.00f,
//                                                      0.00f,
//                                                      0.00f,
//                                                      0.00f};
__device__ __constant__ float2 ss_pixel_offsets_04[4] = {{ss_grid_res * 5.0f, ss_grid_res * 1.0f},
                                                         {ss_grid_res * 7.0f, ss_grid_res * 5.0f},
                                                         {ss_grid_res * 3.0f, ss_grid_res * 7.0f},
                                                         {ss_grid_res * 1.0f, ss_grid_res * 3.0f}};
__device__ __constant__ float ss_sample_offsets_04[4] = {0.00f,
                                                         0.25f,
                                                         0.50f,
                                                         0.75f};
// NV pattern
__device__ __constant__ float2 ss_pixel_offsets_08[8] = {{0.630f, 0.206f},
                                                         {0.667f, 0.079f},
                                                         {0.413f, 0.333f},
                                                         {0.794f, 0.460f},
                                                         {0.032f, 0.587f},
                                                         {0.531f, 0.714f},
                                                         {0.286f, 0.841f},
                                                         {0.921f, 0.968f}};
__device__ __constant__ float ss_sample_offsets_08[8] = {0.000f,
                                                         0.125f,
                                                         0.250f,
                                                         0.375f,
                                                         0.500f,
                                                         0.625f,
                                                         0.750f,
                                                         0.875f};
template<int SAMPLE_COUNT, bool ENABLE_SS>
float2 __device__ __inline__ subpixel_offset(int i) { return make_float2(0.0f); }

template<> __device__ __inline__ float2 subpixel_offset<1, true>(int i)  { return ss_pixel_offsets_01; }
template<> __device__ __inline__ float2 subpixel_offset<1, false>(int i) { return ss_pixel_offsets_01; }
template<> __device__ __inline__ float2 subpixel_offset<4, true>(int i)  { return ss_pixel_offsets_04[i]; }
template<> __device__ __inline__ float2 subpixel_offset<4, false>(int i) { return ss_pixel_offsets_01; }
template<> __device__ __inline__ float2 subpixel_offset<8, true>(int i)  { return ss_pixel_offsets_08[i]; }
template<> __device__ __inline__ float2 subpixel_offset<8, false>(int i) { return ss_pixel_offsets_01; }

template<int SAMPLE_COUNT, bool ENABLE_SS>
float __device__ __inline__ sample_offset(int i) { return 0.0f; }

template<> __device__ __inline__ float sample_offset<1, true>(int i)  { return ss_sample_offsets_01; }
template<> __device__ __inline__ float sample_offset<1, false>(int i) { return ss_sample_offsets_01; }
template<> __device__ __inline__ float sample_offset<4, true>(int i)  { return ss_sample_offsets_04[i]; }
template<> __device__ __inline__ float sample_offset<4, false>(int i) { return ss_sample_offsets_01; }
template<> __device__ __inline__ float sample_offset<8, true>(int i)  { return ss_sample_offsets_08[i]; }
template<> __device__ __inline__ float sample_offset<8, false>(int i) { return ss_sample_offsets_01; }

// cuda globals
surface<void, cudaSurfaceType2D> out_image;

texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> volume_texture;
texture<uchar4,        cudaTextureType1D, cudaReadModeNormalizedFloat> colormap_texture;
//texture<float,  cudaTextureType3D, cudaReadModeElementType> volume_texture;
//texture<float4, cudaTextureType1D, cudaReadModeElementType> colormap_texture;

__device__ __constant__ volume_uniform_data uniform_data;

// helpers
inline __device__ float min(float a, float b) { return (a < b) ? a : b; }
inline __device__ float max(float a, float b) { return (a > b) ? a : b; }
inline __device__ float4 min(float4 a, float4 b) { return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); }
inline __device__ float4 max(float4 a, float4 b) { return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }
inline __device__ float3 min(float3 a, float3 b) { return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
inline __device__ float3 max(float3 a, float3 b) { return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }

inline
__device__ __inline__
float4
mul_matrix4(const float4x4 m, const float4 v)
{
    return make_float4(dot(v, m.rows[0]),  //v), //
                       dot(v, m.rows[1]),  //v), //
                       dot(v, m.rows[2]),  //v), //
                       dot(v, m.rows[3])); //v));//
}                                          
                                           
struct ray                                 
{
    float3  origin;
    float3  direction;
    float3  direction_rec;
}; // struct ray

inline
__device__ __inline__
void
make_ray(struct ray*const r,
         const float2 spos,
         const int2   ssize)
{
    float4 spos_nrm = make_float4((spos.x / (float)ssize.x) * 2.0f - 1.0f,
                                  (spos.y / (float)ssize.y) * 2.0f - 1.0f,
                                  -1.0f,
                                   1.0f);
    //float4 spos_os  = mul_matrix4_ptr(&(vdata->_mvp_matrix_inverse), &spos_nrm);
    float4 spos_os  = mul_matrix4(uniform_data._mvp_matrix_inverse, spos_nrm);
    spos_os /= spos_os.w;

    r->origin        = make_float3(uniform_data._os_camera_position);//.xyz;
    r->direction     = normalize(make_float3(spos_os) - r->origin);//vdata->_mvp_matrix_inverse.s012;//spos_os.xyz;//
    r->direction_rec = 1.0f / r->direction;
}

bool
__device__ __inline__
ray_box_intersection(const struct ray*const r,
                     float3   bbmin,
                     float3   bbmax,
                     float*   tmin,
                     float*   tmax)
{
#if 1
    // compute intersection of ray with all six bbox planes
    float3 tbot = r->direction_rec * (bbmin - r->origin);
    float3 ttop = r->direction_rec * (bbmax - r->origin);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin_a = min(ttop, tbot);
    float3 tmax_a = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    *tmin = max(max(tmin_a.x, tmin_a.y), max(tmin_a.x, tmin_a.z));
    *tmax = min(min(tmax_a.x, tmax_a.y), min(tmax_a.x, tmax_a.z));
#else
    float l1 = (bbmin.x - r->origin.x) * r->direction_rec.x;
    float l2 = (bbmax.x - r->origin.x) * r->direction_rec.x;
    *tmin = min(l1,l2);
    *tmax = max(l1,l2);

    l1   = (bbmin.y - r->origin.y) * r->direction_rec.y;
    l2   = (bbmax.y - r->origin.y) * r->direction_rec.y;
    *tmin = max(min(l1,l2), *tmin);
    *tmax = min(max(l1,l2), *tmax);
        
    l1   = (bbmin.z - r->origin.z) * r->direction_rec.z;
    l2   = (bbmax.z - r->origin.z) * r->direction_rec.z;
    *tmin = max(min(l1,l2), *tmin);
    *tmax = min(max(l1,l2), *tmax);

    //return ((lmax > 0.f) & (lmax >= lmin));
    //return ((lmax > 0.f) & (lmax > lmin));
#endif
    return ((*tmin > 0.0) && (*tmax > *tmin));
}

inline
__device__ __inline__
float
length_sqr(const float3 a, const float3 b)
{
    float3 d = b - a;
    //return mad(d.x, d.x, mad(d.y, d.y, d.z * d.z));
    return dot(d, d);
}

template<bool ENABLED>
struct thread_vis_clock
{
    clock_t _thread_start;
    clock_t _thread_stop;
    
    __device__ __inline__ bool enabled() const {
        return ENABLED;
    }
    __device__ __inline__ void start() {
        _thread_start = clock();
    }
    __device__ __inline__ void stop() {
        _thread_stop = clock();
    }
    __device__ __inline__ float4 pseudo_colored_elapsed() const {
        return tex1D(colormap_texture, (float)(_thread_stop - _thread_start) / 3000000.0f);
    }
};

template<>
struct thread_vis_clock<false>
{
    __device__ __inline__ bool enabled() const { return false; }
    __device__ __inline__ void start() {}
    __device__ __inline__ void stop()  {}
    __device__ __inline__ float4 pseudo_colored_elapsed() const { return make_float4(0.0f); }
};

struct ray_state {
    ray         _ray;       // the sub-pixel ray
    float       _t;
    float4      _cdst;      // destination color 
    float2      _trange;    // the t min/max range of the ray
};

template<render_mode  RENDER_MODE,
         int          SAMPLE_COUNT,
         bool         ENABLE_SS>
struct super_sample_volume_traversal
{
    __device__ __inline__
    float4 traverse_volume(const int2& osize,
                           const int2& opos) const
    {
        return make_float4(0.0f);
    }
};

template<int  SAMPLE_COUNT,
         bool ENABLE_SS>
struct super_sample_volume_traversal<RENDER_SS_RAYS_SEQUENTIALLY_00, SAMPLE_COUNT, ENABLE_SS>
{
    __device__ __inline__
    float4 traverse_volume(const int2& osize,
                           const int2& opos) const
    {
        float4     out_color = make_float4(0.0);
        const int  ss_sample_count = ENABLE_SS ? SAMPLE_COUNT : 1;

        for (int i = 0; i < ss_sample_count; ++i) {
            float tmin = 0.0;
            float tmax = 0.0;
        
            // setup ray 
            struct ray cur_ray;
            const float2 opos_pc =   subpixel_offset<SAMPLE_COUNT, ENABLE_SS>(i)
                                    + make_float2(opos.x, opos.y);
            make_ray(&cur_ray, opos_pc, osize);

            if (ray_box_intersection(&cur_ray, make_float3(0.0), make_float3(uniform_data._volume_extends), &tmin, &tmax)) {
                float3 cam_pos   = make_float3(uniform_data._os_camera_position);
                float3 ray_entry = tmin * cur_ray.direction + cur_ray.origin;
                float3 ray_exit  = tmax * cur_ray.direction + cur_ray.origin;

                float3 ray_increment = cur_ray.direction * uniform_data._sampling_distance.x;
                float3 sampling_pos  = ray_entry + ray_increment; // test, increment just to be sure we are in the volume
                if (ENABLE_SS) {
                    sampling_pos += ray_increment * sample_offset<SAMPLE_COUNT, ENABLE_SS>(i);
                }
                float3 to_tex        = make_float3(uniform_data._scale_obj_to_tex);

                float smpl_sqr_dist  = length_sqr(cam_pos, sampling_pos);
                float exit_sqr_dist  = length_sqr(cam_pos, ray_exit);

                float4 dst = make_float4(0.0f);
                float  opc = uniform_data._sampling_distance.y;

                //out_color = make_float4(ray_exit, 1.0);
                while ((exit_sqr_dist - smpl_sqr_dist) > 0.0f && dst.w < 0.99f) {
                    float3 tc_vol = sampling_pos * to_tex;

                    float  s   = tex3D(volume_texture, tc_vol.x, tc_vol.y, tc_vol.z);// texture(volume_raw, sampling_pos * volume_data.scale_obj_to_tex.xyz).r;
                    float4 src = tex1D(colormap_texture, s);
                    //float4 src    = read_imagef(volume_image, vol_smpl, tc_vol).xxxx;//(float4)(s);//texture(color_map, s);

                    //float4 src = (float4)(s, s, s, 0.1);

                    // increment ray
                    sampling_pos  += ray_increment;
                    smpl_sqr_dist  = length_sqr(cam_pos, sampling_pos);

                    //float3 d = cam_pos - sampling_pos;
                    //smpl_sqr_dist  = dot(d, d);

                    //inside_volume  = inside_volume_bounds(sampling_pos) && (dst.a < 0.99);

                    // opacity correction
                    src.w = 1.0f - pow(1.0f - src.w, opc);

                    // compositing
                    float omda_sa = (1.0 - dst.w) * src.w;
                    dst.x += omda_sa * src.x;
                    dst.y += omda_sa * src.y;
                    dst.z += omda_sa * src.z;
                    dst.w   += omda_sa;
                }
                out_color += dst;
            }
            //else {
            //    out_color += make_float4(1.0f, 0.0f, 0.0f, 1.0f);
            //}
        }

        return out_color / float(ss_sample_count);
    }
}; // RENDER_SS_RAYS_SEQUENTIALLY_00

template<int  SAMPLE_COUNT,
         bool ENABLE_SS>
struct super_sample_volume_traversal<RENDER_SS_RAYS_SEQUENTIALLY_01, SAMPLE_COUNT, ENABLE_SS>
{
    __device__ __inline__
    float4 traverse_volume(const int2& osize,
                           const int2& opos) const
    {
        float4     out_color = make_float4(0.0);
        const int  ss_sample_count = ENABLE_SS ? SAMPLE_COUNT : 1;
        //ray_state  ray_states[ENABLE_SS ? SAMPLE_COUNT : 1];

        for (int i = 0; i < ss_sample_count; ++i) {
            // setup ray
            ray_state r;
            const float2 opos_pc =   subpixel_offset<SAMPLE_COUNT, ENABLE_SS>(i)
                                    + make_float2(opos.x, opos.y);
            make_ray(&(r._ray), opos_pc, osize);
            r._cdst = make_float4(0.0f);

            if (ray_box_intersection(&(r._ray),
                                     make_float3(0.0),
                                     make_float3(uniform_data._volume_extends),
                                     &(r._trange.x),
                                     &(r._trange.y)))
            {
                r._t = r._trange.x + sample_offset<SAMPLE_COUNT, ENABLE_SS>(i) * uniform_data._sampling_distance.x;

                const float3 obj_to_tex  = make_float3(uniform_data._scale_obj_to_tex);
                const float  op_corr     = uniform_data._sampling_distance.y;
                const float  s_dist      = uniform_data._sampling_distance.x;

                while (   r._t < r._trange.y
                       && r._cdst.w < 0.99f)
                {
                    const float3 spos      = r._ray.origin + r._t * r._ray.direction;
                    const float3 vtexcoord = obj_to_tex * spos;

                    const float  s   = tex3D(volume_texture, vtexcoord.x, vtexcoord.y, vtexcoord.z);
                    float4 src       = tex1D(colormap_texture, s);

                    // advance ray
                    r._t += s_dist;

                    // opacity correction
                    src.w = 1.0f - pow(1.0f - src.w, op_corr);

                    // compositing
                    float omda_sa = (1.0 - r._cdst.w) * src.w;
                    r._cdst.x += omda_sa * src.x;
                    r._cdst.y += omda_sa * src.y;
                    r._cdst.z += omda_sa * src.z;
                    r._cdst.w += omda_sa;
                }
                out_color += r._cdst;
            }
        }

        return out_color / float(ss_sample_count);
    }
}; // RENDER_SS_RAYS_SEQUENTIALLY_01

template<int  SAMPLE_COUNT,
         bool ENABLE_SS>
struct super_sample_volume_traversal<RENDER_SS_RAYS_PARALLEL_00, SAMPLE_COUNT, ENABLE_SS>
{
    __device__ __inline__
    float4 traverse_volume(const int2& osize,
                           const int2& opos) const
    {
        float4     out_color        = make_float4(0.0);
        const int  ss_sample_count  = ENABLE_SS ? SAMPLE_COUNT : 1;
        bool       any_ray_running  = false;
        ray_state  ray_states[ENABLE_SS ? SAMPLE_COUNT : 1];
        //ray_state* ray_states = &(sray_states[threadIdx.x][threadIdx.y][0]);

        // setup rays
        #pragma unroll
        for (int i = 0; i < ss_sample_count; ++i) {
            const float2 opos_pc =    subpixel_offset<SAMPLE_COUNT, ENABLE_SS>(i)
                                    + make_float2(opos.x, opos.y);
            make_ray(&(ray_states[i]._ray), opos_pc, osize);

            ray_states[i]._cdst = make_float4(0.0f);

            if (ray_box_intersection(&(ray_states[i]._ray),
                                        make_float3(0.0),
                                        make_float3(uniform_data._volume_extends),
                                        &(ray_states[i]._trange.x),
                                        &(ray_states[i]._trange.y)))
            {
                ray_states[i]._t = ray_states[i]._trange.x + sample_offset<SAMPLE_COUNT, ENABLE_SS>(i) * uniform_data._sampling_distance.x;
                any_ray_running = true;
            }
            else {
                ray_states[i]._t = ray_states[i]._trange.y;
                //ray_states[i]._cdst = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
            }
        }

        const float3& obj_to_tex  = make_float3(uniform_data._scale_obj_to_tex);
        const float&  op_corr     = uniform_data._sampling_distance.y;
        const float&  s_dist      = uniform_data._sampling_distance.x;

        while (any_ray_running) {
            any_ray_running = false;
            
            #pragma unroll
            for (int s = 0; s < ss_sample_count; ++s) {
                ray_state& r = ray_states[s];

                if (   r._t < r._trange.y
                    && r._cdst.w < 0.99f)
                {
                    any_ray_running = true;
                    const float3 spos      = r._ray.origin + r._t * r._ray.direction;
                    const float3 vtexcoord = obj_to_tex * spos;

                    const float  s   = tex3D(volume_texture, vtexcoord.x, vtexcoord.y, vtexcoord.z);
                    float4 src = tex1D(colormap_texture, s);

                    // advance ray
                    r._t += s_dist;

                    // opacity correction
                    src.w = 1.0f - pow(1.0f - src.w, op_corr);

                    // compositing
                    float omda_sa = (1.0 - r._cdst.w) * src.w;
                    r._cdst.x += omda_sa * src.x;
                    r._cdst.y += omda_sa * src.y;
                    r._cdst.z += omda_sa * src.z;
                    r._cdst.w += omda_sa;
                }
            }
        }

        #pragma unroll
        for (int s = 0; s < ss_sample_count; ++s) {
            out_color += ray_states[s]._cdst;
        }

        return out_color / float(ss_sample_count);
    }
}; // RENDER_SS_RAYS_PARALLEL_00


template<render_mode  RENDER_MODE,
         int          SAMPLE_COUNT,
         bool         ENABLE_SS>
void
__global__
main_vrc(unsigned out_image_w, unsigned out_image_h)
{
    thread_vis_clock<false> thread_clock_vis;

    const int2 osize = make_int2(out_image_w, out_image_h);
    const int2 opos  = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y);

    if (opos.x < osize.x && opos.y < osize.y) {
        thread_clock_vis.start();

        super_sample_volume_traversal<RENDER_MODE, SAMPLE_COUNT, ENABLE_SS> ss_vol_trav;
        float4 out_color = ss_vol_trav.traverse_volume(osize, opos);

        thread_clock_vis.stop();
        if (thread_clock_vis.enabled()) {
            out_color = thread_clock_vis.pseudo_colored_elapsed();
        }

        uchar4 out_col_data;
        out_col_data.x = (unsigned char)(out_color.x * 255.0f);
        out_col_data.y = (unsigned char)(out_color.y * 255.0f);
        out_col_data.z = (unsigned char)(out_color.z * 255.0f);
        out_col_data.w = (unsigned char)(out_color.w * 255.0f);

        surf2Dwrite(out_col_data, out_image, opos.x * sizeof(uchar4), opos.y);
    }
}

void
startup_ray_cast_kernel(unsigned out_image_w, unsigned out_image_h,
                        cudaGraphicsResource_t                   output_image_res,
                        cudaGraphicsResource_t                   volume_image_res,
                        cudaGraphicsResource_t                   cmap_image_res,
                        int                                      render_mode,
                        int                                      sample_count,
                        bool                                     use_supersampling,
                        cudaStream_t                             cuda_stream)
{
    cudaError   cu_err = cudaSuccess;

    // output image
    cudaArray*             cu_oi_array = 0;
    cu_err = cudaGraphicsSubResourceGetMappedArray(&cu_oi_array, output_image_res, 0, 0);
    cu_err = cudaBindSurfaceToArray(out_image, cu_oi_array);

    // volume texture
    volume_texture.addressMode[0] = cudaAddressModeClamp;
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;
    volume_texture.filterMode     = cudaFilterModeLinear;
    volume_texture.normalized     = true;
    cudaArray* cu_vi_array = 0;
    cu_err = cudaGraphicsSubResourceGetMappedArray(&cu_vi_array, volume_image_res, 0, 0);
    cu_err = cudaBindTextureToArray(volume_texture, cu_vi_array);

    // color map texture
    colormap_texture.addressMode[0] = cudaAddressModeClamp;
    colormap_texture.filterMode     = cudaFilterModeLinear;
    colormap_texture.normalized     = true;
    cudaArray* cu_ci_array = 0;
    cu_err = cudaGraphicsSubResourceGetMappedArray(&cu_ci_array, cmap_image_res, 0, 0);
    cu_err = cudaBindTextureToArray(colormap_texture, cu_ci_array);

    //cudaFuncSetCacheConfig(main_vrc, cudaFuncCachePreferL1);

    // calculate the grid and block sizes
    //cudaFuncAttributes  cu_krnl_attr;
    //cu_err = cudaFuncGetAttributes(&cu_krnl_attr, main_vrc);

    //std::cout << cu_krnl_attr.maxThreadsPerBlock;

    dim3 vsize = dim3(out_image_w, out_image_h, 1);
    //dim3 bsize = dim3(32, cu_krnl_attr.maxThreadsPerBlock / 32, 1);
    dim3 bsize = dim3(8, 24, 1);
    dim3 gsize;

    gsize.x = vsize.x % bsize.x == 0 ? vsize.x / bsize.x : (vsize.x / bsize.x + 1);
    gsize.y = vsize.y % bsize.y == 0 ? vsize.x / bsize.x : (vsize.y / bsize.y + 1);

    dim3 grid_size(gsize.x, gsize.y, 1);
    dim3 block_size(bsize.x, bsize.y, 1);

    if (use_supersampling) {
        switch (render_mode) {
            case RENDER_SS_RAYS_SEQUENTIALLY_00:
                switch (sample_count) {
                    case 1: main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_00, 1, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                    case 4: main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_00, 4, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                    case 8: main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_00, 8, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                }
            break;
            case RENDER_SS_RAYS_SEQUENTIALLY_01:
                switch (sample_count) {
                    case 1: main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_01, 1, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                    case 4: main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_01, 4, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                    case 8: main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_01, 8, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                }
            break;
            case RENDER_SS_RAYS_PARALLEL_00:
                switch (sample_count) {
                    case 1: main_vrc<RENDER_SS_RAYS_PARALLEL_00, 1, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                    case 4: main_vrc<RENDER_SS_RAYS_PARALLEL_00, 4, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                    case 8: main_vrc<RENDER_SS_RAYS_PARALLEL_00, 8, true><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
                }
            break;
        }
    }
    else {
        switch (render_mode) {
            case RENDER_SS_RAYS_SEQUENTIALLY_00:
                main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_00, 1, false><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
            case RENDER_SS_RAYS_SEQUENTIALLY_01:
                main_vrc<RENDER_SS_RAYS_SEQUENTIALLY_01, 1, false><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
            case RENDER_SS_RAYS_PARALLEL_00:
                main_vrc<RENDER_SS_RAYS_PARALLEL_00, 1, false><<<grid_size, block_size, 0, cuda_stream>>>(out_image_w, out_image_h);break;
        }
    } 
}

bool
upload_uniform_data(const volume_uniform_data& vud,
                    cudaStream_t               cuda_stream)
{
    cudaError cu_err = cudaMemcpyToSymbolAsync(uniform_data, &vud, sizeof(volume_uniform_data), 0, cudaMemcpyHostToDevice, cuda_stream);
    return cudaSuccess == cu_err;
}

} // namespace cuda
} // namespace scm
