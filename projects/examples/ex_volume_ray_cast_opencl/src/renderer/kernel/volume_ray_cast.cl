
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#pragma OPENCL FP_CONTRACT OFF

#include <volume_uniform_data.h>

inline
float4
mul_matrix4(const float4x4 m, const float4 v)
{
#if 1
#if 0
    return (float4)(dot(v, m.s048c),
                    dot(v, m.s159d),
                    dot(v, m.s26ae),
                    dot(v, m.s37bf));
#else
    return (float4)(dot(v, m.s0123),
                    dot(v, m.s4567),
                    dot(v, m.s89ab),
                    dot(v, m.scdef));
#endif
#else
#if 0
    return (float4)(dot(v, (float4)(m.col[0].x, m.col[1].x, m.col[2].x, m.col[3].x)),
                    dot(v, (float4)(m.col[0].y, m.col[1].y, m.col[2].y, m.col[3].y)),
                    dot(v, (float4)(m.col[0].z, m.col[1].z, m.col[2].z, m.col[3].z)),
                    dot(v, (float4)(m.col[0].w, m.col[1].w, m.col[2].w, m.col[3].w)));
#else
    return (float4)(dot(v, m.col[0]),//(float4)(m.col[0].x, m.col[0].y, m.col[0].z, m.col[0].w)),
                    dot(v, m.col[1]),//(float4)(m.col[1].x, m.col[1].y, m.col[1].z, m.col[1].w)),
                    dot(v, m.col[2]),//(float4)(m.col[2].x, m.col[2].y, m.col[2].z, m.col[2].w)),
                    dot(v, m.col[3]));//(float4)(m.col[3].x, m.col[3].y, m.col[3].z, m.col[3].w)));
#endif
#endif
}

struct ray
{
    float3  origin;
    float3  direction;
    float3  direction_rec;
}; // struct ray

inline void
make_ray(struct ray*const r,
         const int2 spos,
         const int2 ssize,
         __constant struct volume_uniform_data* vdata)
{
    float4 spos_nrm = (float4)(((float)spos.x / (float)ssize.x) * 2.0 - 1.0,
                               ((float)spos.y / (float)ssize.y) * 2.0 - 1.0,
                               -1.0,
                                1.0);
    //float4 spos_os  = mul_matrix4_ptr(&(vdata->_mvp_matrix_inverse), &spos_nrm);
    float4 spos_os  = mul_matrix4(vdata->_mvp_matrix_inverse, spos_nrm);
    spos_os.xyz /= spos_os.w;

    r->origin        = vdata->_os_camera_position.xyz;
    r->direction     = normalize(spos_os.xyz - r->origin);//vdata->_mvp_matrix_inverse.s012;//spos_os.xyz;//
    r->direction_rec = 1.0 / r->direction;
}

bool
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
float
length_sqr(const float3 a, const float3 b)
{
    float3 d = b - a;
    //return mad(d.x, d.x, mad(d.y, d.y, d.z * d.z));
    return dot(d, d);
}

__kernel
//__attribute__((work_group_size_hint(32, 32, 0)))
//__attribute__((vec_type_hint(float4)))
void
main_vrc(__write_only image2d_t                        output_image,
         __read_only  image3d_t                        volume_image,
         __read_only  image2d_t                        colormap_image,
         __constant   struct volume_uniform_data*      volume_data)
{
    int2 osize = (int2)(get_image_width(output_image), get_image_height(output_image));
    int2 opos  = (int2)(get_global_id(0), get_global_id(1));


    if (opos.x < osize.x && opos.y < osize.y) {
        float4 out_color;
#if 1
        //float4    tcrd = (float4)(0.0, 0.0, 1.0, 0.0);
        //tcrd.xy = convert_float2(opos.xy) / convert_float2(osize.xy);
        //sampler_t fb_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
        //float4    out_color  = read_imagef(volume_image, fb_sampler, tcrd);

        struct ray cur_ray;
        make_ray(&cur_ray, opos, osize, volume_data);
        //out_color.xyz = (float3)(1.0, 0.0, 0.0);//cur_ray.direction.xyz;//

        float tmin = 0.0;
        float tmax = 0.0;
        
        if (ray_box_intersection(&cur_ray, (float3)(0.0), volume_data->_volume_extends.xyz, &tmin, &tmax)) {
#if 0
            out_color.xyz = (float3)(0.0, 0.0, 1.0);
            out_color.w   = 1.0;
#else
            float3 cam_pos   = volume_data->_os_camera_position.xyz;
            float3 ray_entry = tmin * cur_ray.direction + cur_ray.origin;
            float3 ray_exit  = tmax * cur_ray.direction + cur_ray.origin;

            float3 ray_increment = cur_ray.direction * volume_data->_sampling_distance.x;
            float3 sampling_pos  = ray_entry + ray_increment; // test, increment just to be sure we are in the volume
            float3 to_tex        = volume_data->_scale_obj_to_tex.xyz;

            float smpl_sqr_dist  = length_sqr(cam_pos, sampling_pos);
            float exit_sqr_dist  = length_sqr(cam_pos, ray_exit);

            const sampler_t vol_smpl   = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

            float4 dst = (float4)(0.0);
            float  opc = volume_data->_sampling_distance.y;
            int    loop_count = 0;

#if 1
            while ((exit_sqr_dist - smpl_sqr_dist) > 0.0 && dst.w < 0.99) {
                ++loop_count;
                float4 tc_vol = (float4)(sampling_pos * to_tex, 0.0);
                float  s      = read_imagef(volume_image, vol_smpl, tc_vol).x;// texture(volume_raw, sampling_pos * volume_data.scale_obj_to_tex.xyz).r;
                float2 tc_cm  = (float2)(s, 0.0);
                float4 src    = read_imagef(colormap_image, vol_smpl, tc_cm);//(float4)(s);//texture(color_map, s);
                //float4 src    = read_imagef(volume_image, vol_smpl, tc_vol).xxxx;//(float4)(s);//texture(color_map, s);

                //float4 src = (float4)(s, s, s, 0.1);

                // increment ray
                sampling_pos  += ray_increment;
                smpl_sqr_dist  = length_sqr(cam_pos, sampling_pos);

                //float3 d = cam_pos - sampling_pos;
                //smpl_sqr_dist  = dot(d, d);

                //inside_volume  = inside_volume_bounds(sampling_pos) && (dst.a < 0.99);

                // opacity correction
                src.w = 1.0 - half_powr(1.0 - src.w, opc);

                // compositing
                float omda_sa = (1.0 - dst.w) * src.w;
                dst.xyz += omda_sa * src.xyz;
                dst.w   += omda_sa;
            }

            out_color = dst;
            //out_color = read_imagef(colormap_image, vol_smpl, (float2)((float)(loop_count) / 1500.0, 0.0));//(float4)//dst;
#else
            out_color = (float4)(ray_exit, 1.0);
#endif

#endif
        }
        else {
            out_color = (float4)(0.0);
        }
#else
        struct ray cur_ray;
        make_ray(&cur_ray, opos, osize, volume_data);
        out_color.xyz = cur_ray.direction.xyz;//
        //out_color = (float4)(1.0, 0.0, 0.0, 1.0);
#endif
        //float4    out_color = (float4)(1.0, 1.0, 1.0, 1.0);
        write_imagef(output_image, opos, out_color);//shuffle(out_color, (uint4)(0, 0, 0, 3)));
        //uint4    out_color = (uint)(255, 0, 0, 255);
        //write_imageui(output_image, src_coord, out_color);
    }
}
