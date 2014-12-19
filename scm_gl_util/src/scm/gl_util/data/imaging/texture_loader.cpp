
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_loader.h"

#include <vector>
#include <memory.h>
#include <sstream>

#include <FreeImagePlus.h>

#include <scm/core/math.h>
#include <scm/core/memory.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/data/imaging/texture_image_data.h>

namespace scm {
namespace gl {
namespace {

void scale_colors(float r, float g, float b,
                  int w, int h, data_format format,
                  void* data)
{
    unsigned channels = channel_count(format);

    if (   format == FORMAT_RGB_8
        || format == FORMAT_RGBA_8) {
        unsigned char* cdata = reinterpret_cast<unsigned char*>(data);
        for (unsigned i = 0; i < (w * h * channels); i += channels) {
            cdata[i]       = static_cast<unsigned char>(cdata[i]     * r);
            cdata[i + 1]   = static_cast<unsigned char>(cdata[i + 1] * g);
            cdata[i + 2]   = static_cast<unsigned char>(cdata[i + 2] * b);
        }
    }
    if (   format == FORMAT_BGR_8
        || format == FORMAT_BGRA_8) {
        unsigned char* cdata = reinterpret_cast<unsigned char*>(data);
        for (unsigned i = 0; i < (w * h * channels); i += channels) {
            cdata[i]       = static_cast<unsigned char>(cdata[i]     * b);
            cdata[i + 1]   = static_cast<unsigned char>(cdata[i + 1] * g);
            cdata[i + 2]   = static_cast<unsigned char>(cdata[i + 2] * r);
        }
    }
    if (   format == FORMAT_RGB_32F
        || format == FORMAT_RGBA_32F) {
        float* cdata = reinterpret_cast<float*>(data);
        for (unsigned i = 0; i < (w * h * channels); i += channels) {
            cdata[i]       = cdata[i]     * r;
            cdata[i + 1]   = cdata[i + 1] * g;
            cdata[i + 2]   = cdata[i + 2] * b;
        }
    }
}

std::pair<std::vector<void*>, std::vector<shared_array<unsigned char> > > get_data(const std::string&   in_image_path,
                            bool                 in_create_mips,
                            bool                 in_color_mips,
                            const data_format    in_force_internal_format,
                            math::vec2ui&        out_image_size,
                            data_format&         out_image_format,
                            data_format&         out_image_internal_format,
                            unsigned&            out_num_mipmaps) {

    scm::scoped_ptr<fipImage>   in_image(new fipImage);

    if (!in_image->load(in_image_path.c_str())) {
        glerr() << log::error << "texture_loader::load_texture_2d(): "
                << "unable to open file: " << in_image_path << log::end;
        return {};
    }

    FREE_IMAGE_TYPE  image_type = in_image->getImageType();
    unsigned         image_bit_count = in_image->getInfoHeader()->biBitCount;
    out_image_size = math::vec2ui(in_image->getWidth(), in_image->getHeight());
    //int             image_pitch     = in_image->getScanWidth();
    
    //glout() << log::info << "bitcount " << image_bit_count << " scan width " << static_cast<float>(image_pitch) / 3;

    switch (image_type) {
        case FIT_BITMAP: {
            unsigned num_components = in_image->getBitsPerPixel() / 8;
            switch (num_components) {
                case 1: out_image_format = out_image_internal_format = FORMAT_R_8; break;
                case 2: out_image_format = out_image_internal_format = FORMAT_RG_8; break;
                case 3: out_image_format = FORMAT_BGR_8; out_image_internal_format = FORMAT_RGB_8; break;
                case 4: out_image_format = FORMAT_BGRA_8; out_image_internal_format = FORMAT_RGBA_8; break;
            }
        } break;
        case FIT_INT16:     out_image_format = out_image_internal_format = FORMAT_R_16S; break;
        case FIT_UINT16:    out_image_format = out_image_internal_format = FORMAT_R_16; break;
        case FIT_RGB16:     out_image_format = out_image_internal_format = FORMAT_RGB_16; break;
        case FIT_RGBA16:    out_image_format = out_image_internal_format = FORMAT_RGBA_16; break;
        case FIT_INT32:     break; 
        case FIT_UINT32:    break;
        case FIT_FLOAT:     out_image_format = out_image_internal_format = FORMAT_R_32F; break;
        case FIT_RGBF:      out_image_format = out_image_internal_format = FORMAT_RGB_32F; break;
        case FIT_RGBAF:     out_image_format = out_image_internal_format = FORMAT_RGBA_32F; break;
    }

    if (out_image_format == FORMAT_NULL) {
        glerr() << log::error << "texture_loader::load_texture_2d(): "
                << "unsupported color format: " << std::hex << in_image->getImageType() << log::end;
        return {};
    }

    std::vector<shared_array<unsigned char> >   image_mip_data;
    std::vector<void*>                          image_mip_data_raw;

    if (in_create_mips) {
        out_num_mipmaps = util::max_mip_levels(out_image_size);
    }

    for (unsigned i = 0; i < out_num_mipmaps; ++i) {
        scm::size_t  cur_data_size = 0;
        math::vec2ui lev_size = util::mip_level_dimensions(out_image_size, i);

        if (i == 0) {
            lev_size      = out_image_size;
            cur_data_size =   out_image_size.x * out_image_size.y;
            cur_data_size *=  channel_count(out_image_format);
            cur_data_size *=  size_of_channel(out_image_format);
        }
        else {
            cur_data_size =   lev_size.x * lev_size.y;
            cur_data_size *=  channel_count(out_image_format);
            cur_data_size *=  size_of_channel(out_image_format);

            if (FALSE == in_image->rescale(lev_size.x, lev_size.y, FILTER_LANCZOS3)) {
                glerr() << log::error << "texture_loader::load_texture_2d(): "
                        << "unable to scale image (level: " << i << ", dim: " << lev_size << ")" << log::end;
                return {};
            }
            //image_pitch     = in_image->getScanWidth();
    
            ////glout() << log::info << "bitcount " << image_bit_count << " scan width " << static_cast<float>(image_pitch) / 3;

            ////std::stringstream ofn;
            ////ofn << in_image_path << "_" << i << ".png";
            ////in_image->save(ofn.str().c_str());

            if (in_image->getWidth() != lev_size.x || in_image->getHeight() != lev_size.y) {
                glerr() << log::error << "texture_loader::load_texture_2d(): "
                        << "image dimensions changed after resamling (level: " << i
                        << ", dim: " << lev_size 
                        << ", type: " << std::hex << in_image->getImageType() << ")" << log::end;
                return {};
            }
            if (in_image->getInfoHeader()->biBitCount != image_bit_count) {
                glerr() << log::error << "texture_loader::load_texture_2d(): "
                        << "image bitcount changed after resamling (level: " << i
                        << ", bit_count: " << image_bit_count 
                        << ", img_bit_count: " << in_image->getInfoHeader()->biBitCount << ")" << log::end;
                return {};
            }
            if (image_type != in_image->getImageType()) {
                glerr() << log::error << "texture_loader::load_texture_2d(): "
                        << "image type changed after resamling (level: " << i
                        << ", dim: " << lev_size 
                        << ", type: " << std::hex << in_image->getImageType() << ")" << log::end;
                return {};
            }
        }

        scm::shared_array<unsigned char> cur_data(new unsigned char[cur_data_size]);

        size_t line_pitch = in_image->getScanWidth();
        for (unsigned l = 0; l < lev_size.y; ++l) {
            size_t ls = static_cast<size_t>(lev_size.x) * size_of_format(out_image_format);
            uint8* s =   reinterpret_cast<uint8*>(in_image->accessPixels())
                       + line_pitch * l;
            uint8* d =   reinterpret_cast<uint8*>(cur_data.get())
                       + ls * l;
            memcpy(d, s, ls);
        }

        //if (memcpy(cur_data.get(), in_image->accessPixels(), cur_data_size) != cur_data.get()) {
        //    glerr() << log::error << "texture_loader::load_texture_2d(): "
        //            << "unable to copy image data (level: " << i << ", size: " << cur_data_size << "byte)" << log::end;
        //    return (texture_2d_ptr());
        //}
        if (0 != i && in_color_mips) {
            if      (i % 6 == 1) scale_colors(1, 0, 0, lev_size.x, lev_size.y, out_image_format, cur_data.get());
            else if (i % 6 == 2) scale_colors(0, 1, 0, lev_size.x, lev_size.y, out_image_format, cur_data.get());
            else if (i % 6 == 3) scale_colors(0, 0, 1, lev_size.x, lev_size.y, out_image_format, cur_data.get());
            else if (i % 6 == 4) scale_colors(1, 0, 1, lev_size.x, lev_size.y, out_image_format, cur_data.get());
            else if (i % 6 == 5) scale_colors(0, 1, 1, lev_size.x, lev_size.y, out_image_format, cur_data.get());
            else if (i % 6 == 0) scale_colors(1, 1, 0, lev_size.x, lev_size.y, out_image_format, cur_data.get());
        }

        image_mip_data.push_back(cur_data);
        image_mip_data_raw.push_back(cur_data.get());
    }

    if (in_force_internal_format != FORMAT_NULL) {
        out_image_internal_format = in_force_internal_format;
    }

    return {image_mip_data_raw, image_mip_data};
}

} // namespace

texture_2d_ptr
texture_loader::load_texture_2d(render_device&       in_device,
                                const std::string&   in_image_path,
                                bool                 in_create_mips,
                                bool                 in_color_mips,
                                const data_format    in_force_internal_format)
{

    math::vec2ui    image_size(0,0);
    data_format     image_format = FORMAT_NULL;
    data_format     internal_format = FORMAT_NULL;
    unsigned        num_mip_levels = 1;

    auto data(get_data(in_image_path, in_create_mips, in_color_mips, 
                       in_force_internal_format, image_size, image_format, 
                       internal_format, num_mip_levels));
    
    texture_2d_ptr new_tex = in_device.create_texture_2d(image_size, internal_format, num_mip_levels, 1, 1,
                                                         image_format, data.first);

    if (!new_tex) {
        glerr() << log::error << "texture_loader::load_texture_2d(): "
                << "unable to create texture object (file: " << in_image_path << ")" << log::end;
    }

    data.first.clear();
    data.second.clear();

    return (new_tex);
}

texture_cube_ptr
texture_loader::load_texture_cube(render_device&       in_device,
                                  const std::string&   in_image_path_px,
                                  const std::string&   in_image_path_nx,
                                  const std::string&   in_image_path_py,
                                  const std::string&   in_image_path_ny,
                                  const std::string&   in_image_path_pz,
                                  const std::string&   in_image_path_nz,
                                  bool                 in_create_mips,
                                  bool                 in_color_mips,
                                  const data_format    in_force_internal_format)
{
    math::vec2ui    image_size(0,0);
    data_format     image_format = FORMAT_NULL;
    data_format     internal_format = FORMAT_NULL;
    unsigned        num_mip_levels = 1;

    math::vec2ui    tmp_image_size(0,0);
    data_format     tmp_image_format = FORMAT_NULL;
    data_format     tmp_internal_format = FORMAT_NULL;
    unsigned        tmp_num_mip_levels = 1;

    bool formats_match(true);

    auto data_px(get_data(in_image_path_px, in_create_mips, in_color_mips, 
                          in_force_internal_format, image_size, image_format, 
                          internal_format, num_mip_levels));


    auto data_nx(get_data(in_image_path_nx, in_create_mips, in_color_mips, 
                          in_force_internal_format, tmp_image_size, tmp_image_format, 
                          tmp_internal_format, tmp_num_mip_levels));
    if (tmp_image_size != image_size || tmp_image_format != image_format || tmp_internal_format != internal_format || tmp_num_mip_levels != num_mip_levels) {
        formats_match = false;
    }


    auto data_py(get_data(in_image_path_py, in_create_mips, in_color_mips, 
                          in_force_internal_format, tmp_image_size, tmp_image_format, 
                          tmp_internal_format, tmp_num_mip_levels));
    if (tmp_image_size != image_size || tmp_image_format != image_format || tmp_internal_format != internal_format || tmp_num_mip_levels != num_mip_levels) {
        formats_match = false;
    }


    auto data_ny(get_data(in_image_path_ny, in_create_mips, in_color_mips, 
                          in_force_internal_format, tmp_image_size, tmp_image_format, 
                          tmp_internal_format, tmp_num_mip_levels));
    if (tmp_image_size != image_size || tmp_image_format != image_format || tmp_internal_format != internal_format || tmp_num_mip_levels != num_mip_levels) {
        formats_match = false;
    }


    auto data_pz(get_data(in_image_path_pz, in_create_mips, in_color_mips, 
                          in_force_internal_format, tmp_image_size, tmp_image_format, 
                          tmp_internal_format, tmp_num_mip_levels));
    if (tmp_image_size != image_size || tmp_image_format != image_format || tmp_internal_format != internal_format || tmp_num_mip_levels != num_mip_levels) {
        formats_match = false;
    }


    auto data_nz(get_data(in_image_path_nz, in_create_mips, in_color_mips, 
                          in_force_internal_format, tmp_image_size, tmp_image_format, 
                          tmp_internal_format, tmp_num_mip_levels));
    if (tmp_image_size != image_size || tmp_image_format != image_format || tmp_internal_format != internal_format || tmp_num_mip_levels != num_mip_levels) {
        formats_match = false;
    }

    texture_cube_ptr new_tex;
    
    if (formats_match) {
        new_tex = in_device.create_texture_cube(image_size, internal_format, num_mip_levels,
                                                image_format, data_px.first, data_nx.first, data_py.first, data_ny.first, data_pz.first, data_nz.first);
        if (!new_tex) {
            glerr() << log::error << "texture_loader::load_texture_cube(): "
                    << "unable to create texture object (file: " << in_image_path_px << ")" << log::end;
        }
    } else {
        glerr() << log::error << "texture_loader::load_texture_cube(): "
                << "unable to create cube map object (file: " << in_image_path_px << "): all six textures must have same format" << log::end;
    }

    data_px.first.clear();
    data_px.second.clear();

    data_nx.first.clear();
    data_nx.second.clear();

    data_py.first.clear();
    data_py.second.clear();

    data_ny.first.clear();
    data_ny.second.clear();

    data_pz.first.clear();
    data_pz.second.clear();

    data_nz.first.clear();
    data_nz.second.clear();

    return (new_tex);
}


bool
texture_loader::load_texture_image(const render_device_ptr& in_device,
                                   const texture_2d_ptr&    in_texture,
                                   const std::string&       in_image_path,
                                   const texture_region&    in_region,
                                   const unsigned           in_level)
{
    using namespace scm::gl;
    using namespace scm::math;

    scm::scoped_ptr<fipImage>   in_image(new fipImage);

    if (!in_image->load(in_image_path.c_str())) {
        glerr() << log::error << "texture_loader::load_texture_image(): "
                << "unable to open file: " << in_image_path << log::end;
        return (false);
    }

    FREE_IMAGE_TYPE image_type = in_image->getImageType();
    math::vec2ui    image_size(in_image->getWidth(), in_image->getHeight());
    data_format     image_format = FORMAT_NULL;
    
    switch (image_type) {
        case FIT_BITMAP: {
            unsigned num_components = in_image->getBitsPerPixel() / 8;
            switch (num_components) {
                case 1: image_format = FORMAT_R_8; break;
                case 2: image_format = FORMAT_RG_8; break;
                case 3: image_format = FORMAT_BGR_8; FORMAT_RGB_8; break;
                case 4: image_format = FORMAT_BGRA_8; FORMAT_RGBA_8; break;
            }
        } break;
        case FIT_INT16:     image_format = FORMAT_R_16S; break;
        case FIT_UINT16:    image_format = FORMAT_R_16; break;
        case FIT_RGB16:     image_format = FORMAT_RGB_16; break;
        case FIT_RGBA16:    image_format = FORMAT_RGBA_16; break;
        case FIT_INT32:     break; 
        case FIT_UINT32:    break;
        case FIT_FLOAT:     image_format = FORMAT_R_32F; break;
        case FIT_RGBF:      image_format = FORMAT_RGB_32F; break;
        case FIT_RGBAF:     image_format = FORMAT_RGBA_32F; break;
    }

    if (image_format == FORMAT_NULL) {
        glerr() << log::error << "texture_loader::load_texture_image(): "
                << "unsupported color format: " << std::hex << in_image->getImageType() << log::end;
        return (false);
    }

    if (in_region._dimensions.z != 1) {
        glerr() << log::error << "texture_loader::load_texture_image(): "
                << "requested volume region update (dimensions.z > 1), only 2d images supported." << log::end;
        return (false);
    }

    texture_region update_region(in_region);
    if (   (image_size.x != in_region._dimensions.x)
        || (image_size.y != in_region._dimensions.y)) {
        glout() << log::warning << "texture_loader::load_texture_image(): "
                << "updated region size differs from image dimensions (image cropped or partially updated)." << log::end;
        vec2ui min_size;
        min_size.x = min<unsigned>(image_size.x, in_region._dimensions.x);
        min_size.y = min<unsigned>(image_size.y, in_region._dimensions.y);
        
        if (   (min_size.x < image_size.x)
            || (min_size.y < image_size.y)) {
            in_image->crop(0, 0, min_size.x, min_size.y);
        }
        update_region._dimensions = min_size;
    }

    render_context_ptr context = in_device->main_context();

    if (!context->update_sub_texture(in_texture, update_region, in_level, image_format, in_image->accessPixels()))
    {
        glerr() << log::error << "texture_loader::load_texture_image(): "
                << "error updating texture sub region"
                << "(origin: " << update_region._origin
                << ", dimensions: " << update_region._dimensions << ")." << log::end;
        return (false);
    }

    return (true);
}

texture_image_data_ptr
texture_loader::load_image_data(const std::string&  in_image_path)
{
    scm::scoped_ptr<fipImage>   in_image(new fipImage);

    if (!in_image->load(in_image_path.c_str())) {
        glerr() << log::error << "texture_loader::load_image_data(): "
                << "unable to open file: " << in_image_path << log::end;
        return (texture_image_data_ptr());
    }

    FREE_IMAGE_TYPE image_type = in_image->getImageType();
    math::vec2ui    image_size(in_image->getWidth(), in_image->getHeight());
    data_format     image_format = FORMAT_NULL;
    
    switch (image_type) {
        case FIT_BITMAP: {
            unsigned num_components = in_image->getBitsPerPixel() / 8;
            switch (num_components) {
                case 1: image_format = FORMAT_R_8; break;
                case 2: image_format = FORMAT_RG_8; break;
                case 3: image_format = FORMAT_BGR_8; FORMAT_RGB_8; break;
                case 4: image_format = FORMAT_BGRA_8; FORMAT_RGBA_8; break;
            }
        } break;
        case FIT_INT16:     image_format = FORMAT_R_16S; break;
        case FIT_UINT16:    image_format = FORMAT_R_16; break;
        case FIT_RGB16:     image_format = FORMAT_RGB_16; break;
        case FIT_RGBA16:    image_format = FORMAT_RGBA_16; break;
        case FIT_INT32:     break;
        case FIT_UINT32:    break;
        case FIT_FLOAT:     image_format = FORMAT_R_32F; break;
        case FIT_RGBF:      image_format = FORMAT_RGB_32F; break;
        case FIT_RGBAF:     image_format = FORMAT_RGBA_32F; break;
    }

    if (image_format == FORMAT_NULL) {
        glerr() << log::error << "texture_loader::load_image_data(): "
                << "unsupported color format: " << std::hex << in_image->getImageType() << log::end;
        return (texture_image_data_ptr());
    }

    scm::size_t                 image_data_size = static_cast<size_t>(image_size.x) * image_size.y * size_of_format(image_format);
    scm::shared_array<uint8>    image_data(new uint8[image_data_size]);

    if (memcpy(image_data.get(), in_image->accessPixels(), image_data_size) != image_data.get()) {
        glerr() << log::error << "texture_loader::load_image_data(): "
                << "unable to copy image data." << log::end;
        return (texture_image_data_ptr());
    }

    texture_image_data::level_vector    mip_vec;
    mip_vec.push_back(texture_image_data::level(math::vec3ui(image_size, 1), image_data));

    texture_image_data_ptr ret_data(new texture_image_data(texture_image_data::ORIGIN_LOWER_LEFT, image_format, mip_vec));
    
    return (ret_data);
}

} // namespace gl
} // namespace scm

