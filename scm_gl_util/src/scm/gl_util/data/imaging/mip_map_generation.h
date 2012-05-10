
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_MIP_MAP_GENERATION_H_INCLUDED
#define SCM_GL_UTIL_MIP_MAP_GENERATION_H_INCLUDED

#include <boost/numeric/conversion/bounds.hpp>

namespace scm {
namespace gl {
namespace util {

template<typename vtype,
         const unsigned vdim,
         const int kdim>
void
typed_generate_mipmaps(const math::vec3ui&        src_dim,
                             uint8*               src_data,
                             std::vector<uint8*>& dst_data)
{
    // for non-power of two downsampling using http://developer.nvidia.com/content/non-power-two-mipmapping

    using namespace scm::gl;
    using namespace scm::math;

    const vtype vmax = boost::numeric::bounds<vtype>::highest();//(std::numeric_limits<vtype>::max)();
    const vtype vmin = boost::numeric::bounds<vtype>::lowest();//(std::numeric_limits<vtype>::min)();

    typedef math::vec<vtype, vdim> varr;
    typedef math::vec<float, vdim> tarr;
    tarr zero_arr = tarr(0.0f);
    //std::fill(zero_arr.begin(), zero_arr.end(), size_t(0));
    //for (int i = 0; i < vdim; ++i) zero_arr[i] = 0;

    dst_data.push_back(src_data);

    const int y_max_lines = 3;
    const int z_max_lines = 3;

    for (int l = 1; l < static_cast<int>(util::max_mip_levels(src_dim)); ++l) {
        const vec3i  lsize  = vec3i(util::mip_level_dimensions(src_dim, l));
        const size_t ldsize = static_cast<size_t>(lsize.x) * static_cast<size_t>(lsize.y) * static_cast<size_t>(lsize.z);
        const vec3i  slsize = vec3i(util::mip_level_dimensions(src_dim, l - 1));

        uint8* lrawdata = new uint8[ldsize * sizeof(varr)];
        varr*  ldata    = reinterpret_cast<varr*>(lrawdata);

        scoped_array<tarr>  tlines(new tarr[lsize.x * y_max_lines * z_max_lines]);

        const size_t x_samples = min(slsize.x, (slsize.x & 1) ? 3 : 2);
        const size_t y_samples = min(slsize.y, (slsize.y & 1) ? 3 : 2);
        const size_t z_samples = min(slsize.z, (slsize.z & 1) ? 3 : 2);

        for (int z = 0; z < lsize.z; ++z) {
            for (int y = 0; y < lsize.y; ++y) {
                const varr*  sldata  = reinterpret_cast<varr*>(dst_data[l - 1]);
                {// clear lines
                    memset(tlines.get(), 0, lsize.x * y_max_lines * z_max_lines * sizeof(tarr));
                }
                { // read and sample x-lines
                    if (x_samples == 1) { // 
                        for (int zs = 0; zs < z_samples; ++zs) {
                            for (int ys = 0; ys < y_samples; ++ys) {
                                const varr* ld  = sldata + ( static_cast<size_t>(2 * y + ys) * slsize.x
                                                           + static_cast<size_t>(2 * z + zs) * slsize.x * slsize.y);
                                tlines[(ys + zs * y_max_lines) * lsize.x] = ld[0];
                            }
                        }
                    }
                    else if (x_samples == 2) { // box filter
                        for (int zs = 0; zs < z_samples; ++zs) {
                            for (int ys = 0; ys < y_samples; ++ys) {
                                const varr* ld  = sldata + ( static_cast<size_t>(2 * y + ys) * slsize.x
                                                           + static_cast<size_t>(2 * z + zs) * slsize.x * slsize.y);
                                const int   lo  = (ys + zs * y_max_lines) * lsize.x;
                                for (int x = 0; x < lsize.x; ++x) {
                                    tlines[lo + x] += ld[0];
                                    tlines[lo + x] += ld[1];
                                    tlines[lo + x] *= 0.5f;
                                    ld += 2;
                                }
                            }
                        }
                    }
                    else { // x_samples == 3 ==> polyphase box filter
                        for (int zs = 0; zs < z_samples; ++zs) {
                            for (int ys = 0; ys < y_samples; ++ys) {
                                const varr* ld    = sldata + ( static_cast<size_t>(2 * y + ys) * slsize.x
                                                             + static_cast<size_t>(2 * z + zs) * slsize.x * slsize.y);
                                const int   lo    = (ys + zs * y_max_lines) * lsize.x;
                                const float scale = 1.0f / (2.0f * lsize.x + 1.0f);
                                for (int x = 0; x < lsize.x; ++x) {
                                    const float w0 = static_cast<float>(lsize.x - x);
                                    const float w1 = static_cast<float>(lsize.x);
                                    const float w2 = static_cast<float>(1 + x);

                                    tlines[lo + x] += w0 * tarr(ld[0]); //TODO fix cast
                                    tlines[lo + x] += w1 * tarr(ld[1]);
                                    tlines[lo + x] += w2 * tarr(ld[2]);
                                    tlines[lo + x] *= scale;
                                    ld += 2;
                                }
                            }
                        }
                    }
                }
                { // downsample y-lines
                    if (y_samples == 1) { // nothing to do
                    }
                    else if (y_samples == 2) { // box filter
                        for (int zs = 0; zs < z_samples; ++zs) {
                            const int lo = (zs * y_max_lines) * lsize.x;
                            for (int x = 0; x < lsize.x; ++x) {
                                tlines[lo + x] += tlines[lo + lsize.x + x];
                                tlines[lo + x] *= 0.5f;
                            }
                        }
                    }
                    else { // y_samples == 3 ==> polyphase box filter
                        const float w0 = float(lsize.y - y);
                        const float w1 = float(lsize.y);
                        const float w2 = float(1 + y);
                        for (int zs = 0; zs < z_samples; ++zs) {
                            const int lo      = (zs * y_max_lines) * lsize.x;
                            const float scale = 1.0f / (2.0f * lsize.y + 1.0f);
                            for (int x = 0; x < lsize.x; ++x) {
                                tlines[lo + x]  = w0 * tlines[lo +               x];
                                tlines[lo + x] += w1 * tlines[lo +     lsize.x + x];
                                tlines[lo + x] += w2 * tlines[lo + 2 * lsize.x + x];
                                tlines[lo + x] *= scale;
                            }
                        }
                    }
                }
                { // downsample z-lines
                    if (z_samples == 1) { // nothing to do
                    }
                    else if (z_samples == 2) { // box filter
                        const int lo1 = y_max_lines * lsize.x;
                        for (int x = 0; x < lsize.x; ++x) {
                            tlines[x] += tlines[lo1 + x];
                            tlines[x] *= 0.5f;
                        }
                    }
                    else { // z_samples == 3 ==> polyphase box filter
                        const float w0 = float(lsize.z - z);
                        const float w1 = float(lsize.z);
                        const float w2 = float(1 + z);

                        const int lo1  =     y_max_lines * lsize.x;
                        const int lo2  = 2 * y_max_lines * lsize.x;
                        const float scale = 1.0f / (2.0f * lsize.z + 1.0f);

                        for (int x = 0; x < lsize.x; ++x) {
                            tlines[x]  = w0 * tlines[      x];
                            tlines[x] += w1 * tlines[lo1 + x];
                            tlines[x] += w2 * tlines[lo2 + x];
                            tlines[x] *= scale;
                        }
                    }
                }
                { // write out samples
                    const size_t dst_off =   static_cast<size_t>(y) * lsize.x
                                           + static_cast<size_t>(z) * lsize.x * lsize.y;
                    for (int x = 0; x < lsize.x; ++x) {
                        ldata[dst_off + x] = varr(clamp(tlines[x], tarr(vmin), tarr(vmax)));
                    }
                }
            }
        }

        dst_data.push_back(lrawdata);
    }
}

} // namespace util
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_MIP_MAP_GENERATION_H_INCLUDED
