
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "font_face.h"

#include <iostream>
#include <exception>
#include <stdexcept>
#include <set>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/assign/std/vector.hpp>
//#include <boost/tuple/tuple.hpp>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/font/detail/freetype_types.h>

namespace scm {
namespace gl {

namespace detail {

bool
check_file(const std::string& file_name)
{
    using namespace boost::filesystem;

    path file_path = path(file_name);
    if (!exists(file_path)) {
        return (false);
    }
    if (is_directory(file_path)) {
        return (false);
    }

    return (true);
}

void
find_font_style_files(const std::string&        in_regular_font_file,
                      std::vector<std::string>& out_font_style_files)
{
    using namespace std;
    using namespace boost::assign;
    using namespace boost::filesystem;

    out_font_style_files.clear();
    out_font_style_files.resize(font_face::style_count);

    path    font_file_path = path(in_regular_font_file);
    string  font_file_ext  = font_file_path.extension().string();
    string  font_file_base = font_file_path.stem().string();
    path    font_file_dir  = font_file_path.parent_path();

    { // find and insert regular style
        // TODO: get it working with unix style font file names
        //vector<string> r_prefixes = list_of("")("-R")("L");
        //vector<string> extensions = list_of(".ttf")(".fon");

        out_font_style_files[font_face::style_regular] = in_regular_font_file;
        
        //font_file_path = path(in_regular_font_file);
        //font_file_ext  = font_file_path.extension().string();
        //font_file_base = font_file_path.stem().string();
        //font_file_dir  = font_file_path.parent_path();
    }

    {// search for italic style
        vector<string> i_prefixes = list_of("i")("-RI")("-LI");

        for (auto i = i_prefixes.begin(); i != i_prefixes.end() && out_font_style_files[font_face::style_italic].empty(); ++i) {
            path font_file_italic = font_file_dir / (font_file_base + *i + font_file_ext);

            if (exists(font_file_italic) && !is_directory(font_file_italic)) {
                out_font_style_files[font_face::style_italic] = font_file_italic.string();
            }
        }
    }

    {// search for bold style
        vector<string> b_prefixes = list_of("b")("bd")("-B");

        for (auto i = b_prefixes.begin(); i != b_prefixes.end() && out_font_style_files[font_face::style_bold].empty(); ++i) {
            path font_file_bold = font_file_dir / (font_file_base + *i + font_file_ext);
            if (exists(font_file_bold) && !is_directory(font_file_bold)) {
                out_font_style_files[font_face::style_bold] = font_file_bold.string();
            }
        }
    }

    { // search for bold italic style (z or bi name addition)
        vector<string> bi_prefixes = list_of("z")("bi")("-BI");

        for (auto i = bi_prefixes.begin(); i != bi_prefixes.end() && out_font_style_files[font_face::style_bold_italic].empty(); ++i) {
            path font_file_bold_italic = font_file_dir / (font_file_base + *i + font_file_ext);
            if (exists(font_file_bold_italic) && !is_directory(font_file_bold_italic)) {
                out_font_style_files[font_face::style_bold_italic] = font_file_bold_italic.string();
            }
        }
    }
}

unsigned
available_72dpi_size(const std::string& file_name,
                     unsigned           size,
                     unsigned           disp_res)
{
    using namespace boost::filesystem;

    path        font_file = path(file_name);

    unsigned    font_size   = size;
    unsigned    display_res = disp_res;

    detail::ft_library  ft_lib;
    detail::ft_face     ft_font(ft_lib, font_file.string());

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {

        font_size = (font_size * disp_res + 36) / 72;
    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        std::set<int> available_sizes;

        for (int i = 0; i < ft_font.get_face()->num_fixed_sizes; ++i) {
            available_sizes.insert(ft_font.get_face()->available_sizes[i].height);
        }

        if (available_sizes.empty()) {
            //scm::err() << log::error
            //           << "scm::font::face_loader::available_72dpi_size(): "
            //           << "specified font file ('" << font_file.file_string() << "') "
            //           << "contains fixed size font but fails to report available sizes"
            //           << log::end;
            return (0);
        }

        // scale size to our current display resolution
        font_size = (disp_res * font_size + 36) / 72;

        // now find closest matching size
        std::set<int>::const_iterator lower_bound = available_sizes.lower_bound(font_size); // first >=
        std::set<int>::const_iterator upper_bound = available_sizes.upper_bound(font_size); // first >

        if (   upper_bound == available_sizes.end()) {
            font_size = *available_sizes.rbegin();
        }
        else {
            font_size = *lower_bound;
        }
    }
    else {
        //scm::err() << log::error
        //           << "scm::font::face_loader::available_72dpi_size(): "
        //           << "specified font file ('" << font_file.file_string() << "') "
        //           << "contains unsupported face type"
        //           << log::end;
        return (0);
    }

    return (font_size);
}

} // namesapce detail

font_face::font_face(const render_device_ptr& device,
                     const std::string&       font_file,
                     unsigned                 point_size,
                     float                    border_size,
                     smooth_type              smooth_type,
                     unsigned                 display_dpi)
  : _font_styles(style_count)
  , _font_styles_available(style_count)
  , _font_smooth_style(smooth_type)
  , _point_size(point_size)
  , _border_size(static_cast<unsigned>(math::floor(border_size * 64.0f)))
  , _dpi(display_dpi)
{
    using namespace scm::gl;
    using namespace scm::math;

    try {

        detail::ft_library  ft_lib;

        if (!detail::check_file(font_file)) {
            std::ostringstream s;
            s << "font_face::font_face(): "
              << "font file missing or is a directory ('" << font_file << "')";
            throw(std::runtime_error(s.str()));
        }

        unsigned    font_size   = point_size;
        font_size = detail::available_72dpi_size(font_file, point_size, display_dpi);

        if (font_size == 0) {
            std::ostringstream s;
            s << "font_face::font_face(): "
              << "unable to find suitable font size (font: " << font_file << ", requested size: " << point_size << ", dpi: " << display_dpi << ")";
            throw(std::runtime_error(s.str()));
        }

        font_size = (72 * font_size + display_dpi / 2) / display_dpi;

        std::vector<std::string>    font_style_files;
        detail::find_font_style_files(font_file, font_style_files);

        // fill font styles
        math::vec2ui max_glyph_size(0u, 0u); // to store the maximal glyph size over all styles
        for (int i = 0; i < style_count; ++i) {
            _font_styles_available[i] = !font_style_files[i].empty();

            std::string         cur_font_file = _font_styles_available[i] ? font_style_files[i] : font_style_files[0];
            detail::ft_face     ft_font(ft_lib, cur_font_file);

            ft_font.set_size(font_size, display_dpi);

            // calculate kerning information
            _font_styles[i]._kerning_table.resize(boost::extents[max_char][max_char]);
            for (unsigned l = min_char; l < max_char; ++l) {
                for (unsigned r = min_char; r < max_char; ++r) {
                    //if (l == 'T' && r == 'e')
                    //    std::cout << "kerning: " << int(ft_font.get_kerning(l, r)) << std::endl;
                    _font_styles[i]._kerning_table[l][r] = ft_font.get_kerning(l, r);
                }
            }

            // retrieve the maximal bounding box of all glyphs in the face
            vec2f  font_bbox_x;
            vec2f  font_bbox_y;

            if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {
                float   em_size = 1.0f * ft_font.get_face()->units_per_EM;
                float   x_scale = ft_font.get_face()->size->metrics.x_ppem / em_size;
                float   y_scale = ft_font.get_face()->size->metrics.y_ppem / em_size;

                font_bbox_x = vec2f(ft_font.get_face()->bbox.xMin * x_scale,
                                    ft_font.get_face()->bbox.xMax * x_scale);
                font_bbox_y = vec2f(ft_font.get_face()->bbox.yMin * y_scale,
                                    ft_font.get_face()->bbox.yMax * y_scale);

                _font_styles[i]._line_spacing        = static_cast<unsigned>(ceil(ft_font.get_face()->height * y_scale));
                _font_styles[i]._underline_position  = static_cast<int>(round(ft_font.get_face()->underline_position * y_scale));
                _font_styles[i]._underline_thickness = static_cast<unsigned>(round(ft_font.get_face()->underline_thickness * y_scale));


            }
            else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {
                font_bbox_x = vec2f(0.0f, static_cast<float>(ft_font.get_face()->size->metrics.max_advance >> 6));
                font_bbox_y = vec2f(0.0f, static_cast<float>(ft_font.get_face()->size->metrics.height >> 6));

                _font_styles[i]._line_spacing        = static_cast<int>(font_bbox_y.y);
                _font_styles[i]._underline_position  = -1;
                _font_styles[i]._underline_thickness = 1;
            }
            else {
                std::ostringstream s;
                s << "font_face::font_face(): invalid font face flags (not FT_FACE_FLAG_SCALABLE or FT_FACE_FLAG_FIXED_SIZES), "
                  << "(font: " << cur_font_file << ", size: " << point_size << ")";
                throw(std::runtime_error(s.str()));
            }

            vec2ui font_size = vec2ui(static_cast<unsigned>(ceil(font_bbox_x.y) - floor(font_bbox_x.x)),
                                      static_cast<unsigned>(ceil(font_bbox_y.y) - floor(font_bbox_y.x)));
            max_glyph_size.x = max<unsigned>(max_glyph_size.x, font_size.x);
            max_glyph_size.y = max<unsigned>(max_glyph_size.y, font_size.y);

        }
        // end fill font styles

        // generate texture image
        int             glyph_components     = 0;
        int             glyph_bitmap_ycomp   = 1;
        FT_Render_Mode  glyph_render_mode    = FT_RENDER_MODE_NORMAL;
        unsigned        glyph_load_flags     = FT_LOAD_DEFAULT;
        data_format     glyph_texture_format = FORMAT_NULL;

        switch (smooth_type) {
            case smooth_normal: glyph_components     = 2;
                                glyph_render_mode    = FT_RENDER_MODE_NORMAL; //FT_RENDER_MODE_LIGHT; //
                                glyph_load_flags     = FT_LOAD_DEFAULT; //FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_LIGHT; //
                                glyph_texture_format = FORMAT_RG_8;
                                break;
            case smooth_lcd:    glyph_components     = 3;
                                glyph_bitmap_ycomp   = 3;
                                glyph_render_mode    = FT_RENDER_MODE_LCD;
                                glyph_load_flags     = FT_LOAD_TARGET_LCD;//FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_LIGHT;
                                glyph_texture_format = FORMAT_RGB_8;
                                FT_Library_SetLcdFilter(ft_lib.get_lib(), FT_LCD_FILTER_LIGHT);
                                break;
            default:
                std::ostringstream s;
                s << "font_face::font_face(): unsupported smoothing style.";
                throw(std::runtime_error(s.str()));
        }


        for (int i = 0; i < style_count; ++i) {
            _font_styles[i]._glyphs.resize(max_char);
        }

        //typedef vec<unsigned char, 2> glyph_texel; // 2 components (core, border... TO BE DONE!, currently only first used)
        max_glyph_size += math::vec2ui(1u) + 2 * (_border_size >> 6); // space of at least one texel around all glyphs

        int                           grid_size = static_cast<int>(ceil(math::sqrt(static_cast<double>(max_char - min_char))));
        vec3ui                        glyph_texture_dim  = vec3ui(max_glyph_size * grid_size, style_count); // a 16x16 grid of 256 glyphs in 4 layers
        size_t                        glyph_texture_size = static_cast<size_t>(glyph_texture_dim.x) * glyph_texture_dim.y * glyph_texture_dim.z;
        scoped_array<unsigned char>   glyph_texture(new unsigned char[glyph_texture_size * glyph_components]);

        if (_border_size > 0) { // border
            memset(glyph_texture.get(), 0u, glyph_texture_size * glyph_components); // clear to black

            for (int i = 0; i < style_count; ++i) {
                std::string cur_font_file = _font_styles_available[i] ? font_style_files[i] : font_style_files[0];

                detail::ft_face     ft_font(ft_lib, cur_font_file);
                ft_font.set_size(font_size, display_dpi);
                FT_Bitmap           bitmap;

                for (unsigned c = min_char; c < max_char; ++c) {
                    glyph_info& cur_glyph = _font_styles[i]._glyphs[c];
                    FT_Glyph    ft_glyph;

                    ft_font.load_glyph(c, glyph_load_flags);
                    {
                        detail::ft_stroker stroker(ft_lib, _border_size);
                        FT_Error ft_err;

                        ft_err = FT_Get_Glyph(ft_font.get_glyph(), &ft_glyph);
                        if (ft_err) {
                            throw std::runtime_error("font_face::font_face(): error during FT_Get_Glyph");
                        }
                        ft_err = FT_Glyph_Stroke(&ft_glyph, stroker.get_stroker(), true);
                        if (ft_err) {
                            throw std::runtime_error("font_face::font_face(): error during FT_Glyph_Stroke");
                        }
                        ft_err = FT_Glyph_To_Bitmap(&ft_glyph, glyph_render_mode, 0, true);
                        if (ft_err) {
                            throw std::runtime_error("font_face::font_face(): error during FT_Glyph_To_Bitmap");
                        }
                        FT_BitmapGlyph ft_bitmap_glyph = (FT_BitmapGlyph)ft_glyph;
                        bitmap = ft_bitmap_glyph->bitmap;
                    }

                    // calculate the glyphs grid position in the font texture array
                    vec3ui tex_array_dst;
                    tex_array_dst.x = ((c - min_char) % grid_size) * max_glyph_size.x;
                    tex_array_dst.y = glyph_texture_dim.y - (((c - min_char) / grid_size) + 1) * max_glyph_size.y;
                    //tex_array_dst.x = (c & 0x0F) * max_glyph_size.x;
                    //tex_array_dst.y = glyph_texture_dim.y - ((c >> 4) + 1) * max_glyph_size.y;
                    tex_array_dst.z = i;

                    FT_BitmapGlyph ft_bitmap_glyph = (FT_BitmapGlyph)ft_glyph;
                    cur_glyph._box_size         = vec2i(bitmap.width / glyph_bitmap_ycomp, bitmap.rows);
                    cur_glyph._border_bearing   = vec2i(ft_bitmap_glyph->left, ft_bitmap_glyph->top - ft_bitmap_glyph->bitmap.rows);
                    cur_glyph._texture_origin   = vec2f(static_cast<float>(tex_array_dst.x) / glyph_texture_dim.x,
                                                        static_cast<float>(tex_array_dst.y) / glyph_texture_dim.y);
                    cur_glyph._texture_box_size = vec2f(static_cast<float>(cur_glyph._box_size.x) / glyph_texture_dim.x,
                                                        static_cast<float>(cur_glyph._box_size.y) / glyph_texture_dim.y);

                    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {
                        // linearHoriAdvance contains the 16.16 representation of the horizontal advance
                        // horiAdvance contains only the rounded advance which can be off by 1 and
                        // lead to sub styles beeing rendered to narrow
                        cur_glyph._advance          =  FT_CeilFix(ft_font.get_face()->glyph->linearHoriAdvance) >> 16;
                    }
                    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {
                        cur_glyph._advance          = ft_font.get_face()->glyph->metrics.horiAdvance >> 6;
                    }
                    cur_glyph._bearing          = vec2i(   ft_font.get_face()->glyph->metrics.horiBearingX >> 6,
                                                          (ft_font.get_face()->glyph->metrics.horiBearingY >> 6)
                                                        - (ft_font.get_face()->glyph->metrics.height >> 6));
                    FT_Face ff = ft_font.get_face();
                    int ksks = ft_font.get_face()->glyph->lsb_delta;
                    int ksds = ft_font.get_face()->glyph->rsb_delta;

                    // fill texture
                    switch (bitmap.pixel_mode) {
                        case FT_PIXEL_MODE_GRAY:
                            for (unsigned dy = 0; dy < bitmap.rows; ++dy) {
                                unsigned src_off = dy * bitmap.pitch;
                                unsigned dst_off =    tex_array_dst.x
                                                   + (tex_array_dst.y + bitmap.rows - 1 - dy) * glyph_texture_dim.x
                                                   + i * (glyph_texture_dim.x * glyph_texture_dim.y);
                                for (unsigned dx = 0; dx < bitmap.width; ++dx) {
                                    glyph_texture[(dst_off + dx) * glyph_components] = bitmap.buffer[src_off + dx];
                                }
                            }
                            break;
                        case FT_PIXEL_MODE_LCD:
                            for (unsigned dy = 0; dy < bitmap.rows; ++dy) {
                                unsigned src_off = dy * bitmap.pitch;
                                unsigned dst_off =    tex_array_dst.x
                                                   + (tex_array_dst.y + bitmap.rows - 1 - dy) * glyph_texture_dim.x
                                                   + i * (glyph_texture_dim.x * glyph_texture_dim.y);
                                for (unsigned dx = 0; dx < bitmap.width / glyph_bitmap_ycomp; ++dx) {
                                    glyph_texture[(dst_off + dx) * glyph_components    ] = bitmap.buffer[src_off + dx * glyph_bitmap_ycomp];
                                    glyph_texture[(dst_off + dx) * glyph_components + 1] = bitmap.buffer[src_off + dx * glyph_bitmap_ycomp + 1];
                                    glyph_texture[(dst_off + dx) * glyph_components + 2] = bitmap.buffer[src_off + dx * glyph_bitmap_ycomp + 2];
                                }
                            }
                            break;
                        case FT_PIXEL_MODE_MONO:
                            for (unsigned dy = 0; dy < bitmap.rows; ++dy) {
                                for (int dx = 0; dx < bitmap.pitch; ++dx) {
                                    unsigned        src_off     = dx + dy * bitmap.pitch;
                                    unsigned char   src_byte    = bitmap.buffer[src_off];
                                    for (int bx = 0; bx < 8; ++bx) {
                                        unsigned        dst_off =   (tex_array_dst.x + dx * 8 + bx)
                                                                  + (tex_array_dst.y + bitmap.rows - 1 - dy) * glyph_texture_dim.x
                                                                  + i * (glyph_texture_dim.x * glyph_texture_dim.y);
                                        unsigned char   src_set = src_byte & (0x80 >> bx);
                                        unsigned char*  plah    = &src_byte;
                                        for (int l = 0; l < glyph_components; ++l) {
                                            glyph_texture[dst_off * glyph_components + l] = src_set ? 255u : 0u;
                                        }
                                    }
                                }
                            }
                            break;
                        default:
                            continue;
                    }
                    FT_Done_Glyph(ft_glyph);
                }
            }
            // end generate texture image
            std::vector<void*> image_array_data_raw;
            image_array_data_raw.push_back(glyph_texture.get());

            _font_styles_border_texture_array = device->create_texture_2d(vec2ui(glyph_texture_dim.x, glyph_texture_dim.y),
                                                                   glyph_texture_format, 1, glyph_texture_dim.z, 1,
                                                                   glyph_texture_format, image_array_data_raw);

            if (!_font_styles_border_texture_array) {
                std::ostringstream s;
                s << "font_face::font_face(): unable to create texture object (border).";
                throw(std::runtime_error(s.str()));
            }

            image_array_data_raw.clear();
        }
        // TODO the bearing and box is not wrong!!! FIXME
        { // core
            memset(glyph_texture.get(), 0u, glyph_texture_size * glyph_components); // clear to black

            for (int i = 0; i < style_count; ++i) {
                std::string cur_font_file = _font_styles_available[i] ? font_style_files[i] : font_style_files[0];

                detail::ft_face     ft_font(ft_lib, cur_font_file);
                ft_font.set_size(font_size, display_dpi);
                FT_Bitmap           bitmap;

                for (unsigned c = min_char; c < max_char; ++c) {
                    glyph_info&      cur_glyph = _font_styles[i]._glyphs[c];

                    ft_font.load_glyph(c, glyph_load_flags);
                    if (FT_Render_Glyph(ft_font.get_glyph(), glyph_render_mode)) {
                        //glout() << static_cast<char>(c) << " afafaf";
                        continue;
                    }
                    bitmap = ft_font.get_face()->glyph->bitmap;

                    // calculate the glyphs grid position in the font texture array
                    vec3ui tex_array_dst;
                    tex_array_dst.x = ((c - min_char) % grid_size) * max_glyph_size.x;
                    tex_array_dst.y = glyph_texture_dim.y - (((c - min_char) / grid_size) + 1) * max_glyph_size.y;
                    //tex_array_dst.x = (c & 0x0F) * max_glyph_size.x;
                    //tex_array_dst.y = glyph_texture_dim.y - ((c >> 4) + 1) * max_glyph_size.y;
                    tex_array_dst.z = i;

                    vec2i cur_core_box = vec2i(bitmap.width / glyph_bitmap_ycomp, bitmap.rows);
                    vec2i cur_bm_bearing = vec2i(ft_font.get_face()->glyph->bitmap_left, ft_font.get_face()->glyph->bitmap_top - ft_font.get_face()->glyph->bitmap.rows);
                    vec2i box_diff = vec2i::zero();
                    if (_border_size > 0) {
                        box_diff.x = max(0, cur_bm_bearing.x - cur_glyph._border_bearing.x);
                        box_diff.y = max(0, cur_bm_bearing.y - cur_glyph._border_bearing.y);
                    }
                    cur_glyph._box_size.x = max(cur_glyph._box_size.x, cur_core_box.x);
                    cur_glyph._box_size.y = max(cur_glyph._box_size.y, cur_core_box.y);

                    cur_glyph._texture_origin   = vec2f(static_cast<float>(tex_array_dst.x) / glyph_texture_dim.x,
                                                        static_cast<float>(tex_array_dst.y) / glyph_texture_dim.y);
                    cur_glyph._texture_box_size = vec2f(static_cast<float>(cur_glyph._box_size.x) / glyph_texture_dim.x,
                                                        static_cast<float>(cur_glyph._box_size.y) / glyph_texture_dim.y);

                    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {
                        // linearHoriAdvance contains the 16.16 representation of the horizontal advance
                        // horiAdvance contains only the rounded advance which can be off by 1 and
                        // lead to sub styles beeing rendered to narrow
                        cur_glyph._advance          =  FT_CeilFix(ft_font.get_face()->glyph->linearHoriAdvance) >> 16;
                    }
                    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {
                        cur_glyph._advance          = ft_font.get_face()->glyph->metrics.horiAdvance >> 6;
                    }
                    cur_glyph._bearing          = //vec2i(   ft_font.get_face()->glyph->metrics.horiBearingX >> 6,
                                                  //        (ft_font.get_face()->glyph->metrics.horiBearingY >> 6)
                                                  //      - (ft_font.get_face()->glyph->metrics.height >> 6))
                                                    cur_bm_bearing
                                                  - box_diff;
                                                  //+ cur_glyph._border_bearing;
                    // fill texture
                    switch (bitmap.pixel_mode) {
                        case FT_PIXEL_MODE_GRAY:
                            for (unsigned dy = 0; dy < bitmap.rows; ++dy) {
                                unsigned src_off = dy * bitmap.pitch;
                                unsigned dst_off =    tex_array_dst.x + box_diff.x
                                                   + (tex_array_dst.y + box_diff.y + bitmap.rows - 1 - dy) * glyph_texture_dim.x
                                                   + i * (glyph_texture_dim.x * glyph_texture_dim.y);
                                for (unsigned dx = 0; dx < bitmap.width; ++dx) {
                                    glyph_texture[(dst_off + dx) * glyph_components] = bitmap.buffer[src_off + dx];
                                }
                            }
                            break;
                        case FT_PIXEL_MODE_LCD:
                            for (unsigned dy = 0; dy < bitmap.rows; ++dy) {
                                unsigned src_off = dy * bitmap.pitch;
                                unsigned dst_off =    tex_array_dst.x + box_diff.x
                                                   + (tex_array_dst.y + box_diff.y + bitmap.rows - 1 - dy) * glyph_texture_dim.x
                                                   + i * (glyph_texture_dim.x * glyph_texture_dim.y);
                                for (unsigned dx = 0; dx < bitmap.width / glyph_bitmap_ycomp; ++dx) {
                                    glyph_texture[(dst_off + dx) * glyph_components    ] = bitmap.buffer[src_off + dx * glyph_bitmap_ycomp];
                                    glyph_texture[(dst_off + dx) * glyph_components + 1] = bitmap.buffer[src_off + dx * glyph_bitmap_ycomp + 1];
                                    glyph_texture[(dst_off + dx) * glyph_components + 2] = bitmap.buffer[src_off + dx * glyph_bitmap_ycomp + 2];
                                }
                            }
                            break;
                        case FT_PIXEL_MODE_MONO:
                            for (unsigned dy = 0; dy < bitmap.rows; ++dy) {
                                for (int dx = 0; dx < bitmap.pitch; ++dx) {
                                    unsigned        src_off     = dx + dy * bitmap.pitch;
                                    unsigned char   src_byte    = bitmap.buffer[src_off];
                                    for (int bx = 0; bx < 8; ++bx) {
                                        unsigned        dst_off =   (tex_array_dst.x + box_diff.x + dx * 8 + bx)
                                                                  + (tex_array_dst.y + box_diff.y + bitmap.rows - 1 - dy) * glyph_texture_dim.x
                                                                  + i * (glyph_texture_dim.x * glyph_texture_dim.y);
                                        unsigned char   src_set = src_byte & (0x80 >> bx);
                                        unsigned char*  plah    = &src_byte;
                                        for (int l = 0; l < glyph_components; ++l) {
                                            glyph_texture[dst_off * glyph_components + l] = src_set ? 255u : 0u;
                                        }
                                    }
                                }
                            }
                            break;
                        default:
                            continue;
                    }
                }
            }
            // end generate texture image
            std::vector<void*> image_array_data_raw;
            image_array_data_raw.push_back(glyph_texture.get());

            _font_styles_texture_array = device->create_texture_2d(vec2ui(glyph_texture_dim.x, glyph_texture_dim.y),
                                                                   glyph_texture_format, 1, glyph_texture_dim.z, 1,
                                                                   glyph_texture_format, image_array_data_raw);

            if (!_font_styles_texture_array) {
                std::ostringstream s;
                s << "font_face::font_face(): unable to create texture object.";
                throw(std::runtime_error(s.str()));
            }

            image_array_data_raw.clear();
        }

        glyph_texture.reset();

        std::stringstream os;
        os << std::fixed << std::setprecision(2)
           << "font_face::font_face(): " << std::endl
           << " - created font textures for font '" << font_file << "' "
           << "(point size: " << point_size << ", border size: " << border_size << ")" << std::endl
           << "   - style texture:  format " << gl::format_string(glyph_texture_format)
                << ", size " <<      glyph_texture_dim
                << ", glyph box " << max_glyph_size
                << ", memory " <<      static_cast<double>(glyph_texture_dim.x * glyph_texture_dim.y * glyph_texture_dim.z * size_of_format(glyph_texture_format)) / 1024.0 << "KiB";
        if (_border_size > 0) {
            os << std::endl
               << "   - border texture: format " << gl::format_string(glyph_texture_format)
               << ", size " <<      glyph_texture_dim
               << ", glyph box " << max_glyph_size
               << ", memory " <<      static_cast<double>(glyph_texture_dim.x * glyph_texture_dim.y * glyph_texture_dim.z * size_of_format(glyph_texture_format)) / 1024.0 << "KiB";
        }
        glout() << log::info << os.str();

        using namespace boost::filesystem;

        path            font_file_path = path(font_file);
        std::string     font_file_base = font_file_path.stem().string();
        _name = font_file_base;
    }
    catch(...) {
        cleanup();
        throw;
    }
}

font_face::~font_face()
{
    cleanup();
}

const std::string&
font_face::name() const
{
    return (_name);
}

unsigned
font_face::point_size() const
{
    return (_point_size);
}

unsigned
font_face::border_size() const
{
    return (_border_size);
}

unsigned
font_face::dpi() const
{
    return (_dpi);
}

font_face::smooth_type
font_face::smooth_style() const
{
    return (_font_smooth_style);
}

bool
font_face::has_style(style_type s) const
{
    return (_font_styles_available[s]);
}

const font_face::glyph_info&
font_face::glyph(char c, style_type s) const
{
    return (_font_styles[s]._glyphs[c]);
}

unsigned
font_face::line_advance(style_type s) const
{
    return (_font_styles[s]._line_spacing);
}

int
font_face::kerning(char l, char r, style_type s) const
{
    return (_font_styles[s]._kerning_table[l][r]);
}

int
font_face::underline_position(style_type s) const
{
    return (_font_styles[s]._underline_position);
}

int
font_face::underline_thickness(style_type s) const
{
    return (_font_styles[s]._underline_thickness);
}

void
font_face::cleanup()
{
    _font_styles.clear();
    _font_styles_available.clear();
    _font_styles_texture_array.reset();
}

const texture_2d_ptr&
font_face::styles_texture_array() const
{
    return (_font_styles_texture_array);
}

const texture_2d_ptr&
font_face::styles_border_texture_array() const
{
    return (_font_styles_border_texture_array);
}

} // namespace gl
} // namespace scm
