
#include "face_loader.h"

#include <exception>
#include <set>
#include <stdexcept>

#include <scm_core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm_core/utilities/boost_warning_enable.h>

#include <scm_core/console.h>

#include <scm_core/font/face.h>
#include <scm_core/font/detail/freetype_types.h>

using namespace scm::font;

face_loader::face_loader(const std::string& res_path)
  : _resources_path(res_path)
{
}

face_loader::~face_loader()
{
}

bool face_loader::load(const std::string& file_name,
                       unsigned           size,
                       unsigned           disp_res)
{
    using namespace boost::filesystem;

    path        font_file   = path(file_name);
    unsigned    font_size   = size;
    unsigned    display_res = disp_res;

    if (!exists(font_file)) {
        font_file = path(_resources_path) / font_file;

        if (!exists(font_file)) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "face_loader::load(): "
                               << "unable to find specified font file directly ('" << file_name << "') "
                               << "or relative to resource path ('" << font_file.string() << "')"
                               << std::endl;

            return (false);
        }
    }

    if (is_directory(font_file)) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load(): "
                           << "specified file name ('" << file_name << "') "
                           << "is not a file"
                           << std::endl;

        return (false);
    }

    // find font styles ({name}i|b|z.{ext})
    // load font styles

    detail::ft_library  ft_lib;
    detail::ft_face     ft_font;

    if (!ft_lib.open()) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load(): "
                           << "unable to initialize freetype library"
                           << std::endl;

        return (false);
    }

    if (!ft_font.open_face(ft_lib, font_file.file_string())) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load(): "
                           << "unable to load font file ('" << font_file.file_string() << "')"
                           << std::endl;

        return (false);
    }

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {

    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        std::set<int> available_sizes;

        for (int i = 0; i < ft_font.get_face()->num_fixed_sizes; ++i) {
            available_sizes.insert(ft_font.get_face()->available_sizes[i].height);
        }

        if (available_sizes.empty()) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "face_loader::load(): "
                               << "specified font file ('" << font_file.file_string() << "') "
                               << "contains fixed size font but fails to report available sizes"
                               << std::endl;

            return (false);
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

        // ok bitmap fonts are in pixel sizes (i.e. 72dpi)
        // so scale the pixel size to our display resolution
        // for the following functions
        font_size = (72 * font_size + disp_res / 2) / disp_res;
    }
    else {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load(): "
                           << "specified font file ('" << font_file.file_string() << "') "
                           << "contains unsupported face type"
                           << std::endl;

        return (false);
    }

    if (FT_Set_Char_Size(ft_font.get_face(), 0, font_size << 6, 0, disp_res) != 0) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load(): "
                           << "unable to set character size ('" << font_size << "') "
                           << std::endl;

        return (false);
    }

    // calculate the maximal bounding box of all glyphs in the face
    math::vec2f_t font_bbox_x;
    math::vec2f_t font_bbox_y;
    math::vec2i_t font_glyph_bbox_size;

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {
        float   em_size = 1.0f * ft_font.get_face()->units_per_EM;
        float   x_scale = ft_font.get_face()->size->metrics.x_ppem / em_size;
        float   y_scale = ft_font.get_face()->size->metrics.y_ppem / em_size;

        font_bbox_x = math::vec2f_t(ft_font.get_face()->bbox.xMin * x_scale,
                                    ft_font.get_face()->bbox.xMax * x_scale);
        font_bbox_y = math::vec2f_t(ft_font.get_face()->bbox.yMin * y_scale,
                                    ft_font.get_face()->bbox.yMax * y_scale);
    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        font_bbox_x = math::vec2f_t(0.0f,
                                    static_cast<float>(ft_font.get_face()->size->metrics.max_advance >> 6));
        font_bbox_y = math::vec2f_t(0.0f,
                                    static_cast<float>(ft_font.get_face()->size->metrics.height >> 6));
    }

    font_glyph_bbox_size  = math::vec2i_t(static_cast<int>(math::ceil(font_bbox_x.y) - math::floor(font_bbox_x.x)),
                                          static_cast<int>(math::ceil(font_bbox_y.y) - math::floor(font_bbox_y.x)));

    // allocate texture space
    // glyphs are stacked 16x16 in the texture
    math::vec2i_t   font_texture_size = math::vec2i_t(font_glyph_bbox_size * 16);

    // allocate texture destination memory
    boost::scoped_array<unsigned char> font_texture;

    try {
        font_texture.reset(new unsigned char[font_texture_size.x * font_texture_size.y]);
    }
    catch (std::bad_alloc&) {
        font_texture.reset();
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load(): "
                           << "unable to allocate font texture memory of size ('"
                           << font_texture_size.x * font_texture_size.y << "') bytes"
                           << std::endl;

        return (false);
    }

    // clear texture background to black
    memset(font_texture.get(), 0u, font_texture_size.x * font_texture_size.y);


    unsigned dst_x;
    unsigned dst_y;

    //font_face cur_font_face;

    for (unsigned i = 0; i < 256; ++i) {

        if(FT_Load_Glyph(ft_font.get_face(), FT_Get_Char_Index(ft_font.get_face(), i), FT_LOAD_DEFAULT)) {
            continue;
        }

        if (FT_Render_Glyph(ft_font.get_face()->glyph, FT_RENDER_MODE_NORMAL)) {
            continue;
        }
        FT_Bitmap& bitmap = ft_face->glyph->bitmap;

        // calculate the glyphs grid position in the font texture
        dst_x =                   (i & 0x0F)     * font_glyph_size.x;
        dst_y = font_tex_size.y - ((i >> 4) + 1) * font_glyph_size.y;

            math::vec2i_t ftex_size(0, bitmap.rows);

            switch (bitmap.pixel_mode) {
                case FT_PIXEL_MODE_GRAY:

                    ftex_size.x = bitmap.width;

                    //if (bitmap.pitch > font_glyph_size.x) {
                    //    std::cout << i << " " << (bitmap.pitch - font_glyph_size.x) << std::endl;
                    //}

                    for (int dy = 0; dy < bitmap.rows; ++dy) {

                        unsigned src_off = dy * bitmap.pitch;
                        unsigned dst_off = dst_x + (dst_y + bitmap.rows - 1 - dy) * font_tex_size.x;// + (font_glyph_size.y - bitmap.rows) * font_tex_size.x;
                        memcpy(font_tex.get() + dst_off, bitmap.buffer + src_off, bitmap.width);
                    }
                    break;
                case FT_PIXEL_MODE_MONO:
                    //std::cout << "r: " << bitmap.rows << "\tw: " << bitmap.width << "\tp: " << bitmap.pitch << std::endl;
                    ftex_size.x = bitmap.width;
                    for (int dy = 0; dy < bitmap.rows; ++dy) {
                        for (int dx = 0; dx < bitmap.pitch; ++dx) {

                            unsigned        src_off     = dx + dy * bitmap.pitch;
                            unsigned char   src_byte    = bitmap.buffer[src_off];

                            for (int bx = 0; bx < 8; ++bx) {

                                unsigned dst_off    = (dst_x + dx * 8 + bx) + (dst_y + bitmap.rows - 1 - dy) * font_tex_size.x;

                                unsigned char  src_set = src_byte & (0x80 >> bx);
                                unsigned char* plah = &src_byte;

                                font_tex[dst_off] = src_set ? 255u : 0u;
                            }
                        }
                    }
                    break;
                default:
                    std::cout << "unsupported pixel_mode" << std::endl;
                    continue;

            }
            cur_font_face._tex_lower_left   = math::vec2i_t(dst_x, dst_y);
            cur_font_face._tex_upper_right  = cur_font_face._tex_lower_left + ftex_size;

            cur_font_face._hor_advance      = ft_face->glyph->metrics.horiAdvance >> 6;
            cur_font_face._bearing          = math::vec2i_t(ft_face->glyph->metrics.horiBearingX >> 6,
                                                            (ft_face->glyph->metrics.horiBearingY >> 6) - (ft_face->glyph->metrics.height >> 6));

            font_faces.insert(font_face_container::value_type(i, cur_font_face));

            //std::cout << i << " w: " << bitmap.width << "\th: " << bitmap.rows << "\tp: " << bitmap.pitch << std::endl;

            //for (int dy = 0; dy < bitmap.rows; ++dy) {

            //    unsigned src_off = dy * bitmap.width;
            //    unsigned dst_off = dst_x + (dst_y + bitmap.rows - 1 - dy) * font_tex_size.x;
            //    memcpy(font_tex.get() + dst_off, bitmap.buffer + src_off, bitmap.width);
            //}



    //        //std::cout << i << " w: " << bitmap.width << "\th: " << bitmap.rows << "\tax: " << (ft_face->glyph->advance.x >> 6) << "\tay: " << 
    //                           //  (/*float(ft_face->max_advance_height) / float(ft_face->units_per_EM) * */ft_face->size->metrics.height >> 6) << std::endl;//(ft_face->glyph->linearVertAdvance >> 16) << "." << (ft_face->glyph->linearVertAdvance & 0x0000FFFF)<< std::endl;
    //        //std::cout << i << " w: " << (float)(ft_face->bbox.xMax - ft_face->bbox.xMin) / (float)64 << " h: " << (float)(ft_face->bbox.yMax - ft_face->bbox.yMin) / (float)64 << std::endl;


        }
    }

    return (false);
}