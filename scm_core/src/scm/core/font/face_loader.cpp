
#include "face_loader.h"

#include <cassert>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/log.h>

#include <scm/core/font/face.h>
#include <scm/core/font/detail/freetype_types.h>

namespace scm {
namespace font {

face_loader::face_loader()
{
}

face_loader::~face_loader()
{
    free_texture_resources();
}

void face_loader::set_font_resource_path(const std::string& res_path)
{
    _resources_path = res_path;
}

bool face_loader::check_font_file(const std::string& in_file_name,
                                  std::string& out_file_path) const
{
    using namespace boost::filesystem;

    path        font_file   = path(in_file_name);

    if (!exists(font_file)) {
        font_file = path(_resources_path) / font_file;

        if (!exists(font_file)) {
            scm::err() << log::error
                       << "scm::font::face_loader::check_font_file(): "
                       << "unable to find specified font file directly ('" << in_file_name << "') "
                       << "or relative to resource path ('" << font_file.string() << "')"
                       << log::end;
            return (false);
        }
    }

    if (is_directory(font_file)) {
        scm::err() << log::error
                   << "scm::font::face_loader::check_font_file(): "
                   << "specified file name ('" << in_file_name << "') "
                   << "is not a file"
                   << log::end;
        return (false);
    }

    out_file_path = font_file.file_string();

    return (true);
}


bool face_loader::load(face&              font_face,
                       const std::string& file_name,
                       unsigned           size,
                       unsigned           disp_res)
{
    free_texture_resources();
    font_face.clear();

    using namespace boost::filesystem;

    std::string font_file_name;

    if (!check_font_file(file_name, font_file_name)) {
        return (false);
    }

    path        font_file = path(font_file_name);

    unsigned    font_size   = size;
    unsigned    display_res = disp_res;

    // find font styles ({name}i|b|z|bi.{ext})
    typedef std::map<face::style_type, std::string> styles_container;

    styles_container styles;
    find_font_styles(font_file.file_string(), styles);

    // load font styles
    styles_container::const_iterator style_it;

    font_size = available_72dpi_size(font_file.file_string(),
                                     size,
                                     disp_res);

    if (font_size == 0) {
        return (false);
    }

    unsigned font_size_at_disp_res = font_size = (72 * font_size + disp_res / 2) / disp_res;

    for (style_it  = styles.begin();
         style_it != styles.end();
         ++style_it) {

        if (!load_style(style_it->first,
                        style_it->second,
                        font_face,
                        font_size_at_disp_res,
                        disp_res)) {

            scm::err() << log::error
                       << "scm::font::face_loader::load(): "
                       << "error loading face style (id ='" << style_it->first << "') "
                       << "using font file ('" << style_it->second << "')"
                       << log::end;

            free_texture_resources();
            font_face.clear();

            return (false);
        }
    }

    font_face._name             = font_file.file_string();
    font_face._size_at_72dpi    = font_size;

    return (true);
}

bool face_loader::load_style(face::style_type   style,
                             const std::string& file_name,
                             face&              font_face,
                             unsigned           size,
                             unsigned           disp_res)
{
    using namespace scm::math;

    face_style::kerning_table&              cur_kerning_table = font_face._glyph_mappings[style]._kerning_table;
    face_style::character_glyph_mapping&    cur_glyph_mapping = font_face._glyph_mappings[style]._glyph_mapping;
    texture_type&                           cur_texture       = _face_style_textures[style];
    int&                                    cur_uline_pos     = font_face._glyph_mappings[style]._underline_position;
    unsigned&                               cur_uline_thick   = font_face._glyph_mappings[style]._underline_thickness;
    unsigned&                               cur_line_spacing  = font_face._glyph_mappings[style]._line_spacing;

    detail::ft_library  ft_lib;
    detail::ft_face     ft_font;

    if (!ft_lib.open()) {
        scm::err() << log::error
                   << "scm::font::face_loader::load_style(): "
                   << "unable to initialize freetype library"
                   << log::end;

        return (false);
    }

    if (!ft_font.open_face(ft_lib, file_name)) {
        scm::err() << log::error
                   << "scm::font::face_loader::load_style(): "
                   << "unable to load font file ('" << file_name << "')"
                   << log::end;

        return (false);
    }

    if (FT_Set_Char_Size(ft_font.get_face(), 0, size << 6, 0, disp_res) != 0) {
        scm::err() << log::error
                   << "scm::font::face_loader::load_style(): "
                   << "unable to set character size ('" << size << "') "
                   << log::end;

        return (false);
    }

    // calculate the maximal bounding box of all glyphs in the face
    vec2f font_bbox_x;
    vec2f font_bbox_y;
    vec2i font_glyph_bbox_size;

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {
        float   em_size = 1.0f * ft_font.get_face()->units_per_EM;
        float   x_scale = ft_font.get_face()->size->metrics.x_ppem / em_size;
        float   y_scale = ft_font.get_face()->size->metrics.y_ppem / em_size;

        font_bbox_x = vec2f(ft_font.get_face()->bbox.xMin * x_scale,
                            ft_font.get_face()->bbox.xMax * x_scale);
        font_bbox_y = vec2f(ft_font.get_face()->bbox.yMin * y_scale,
                            ft_font.get_face()->bbox.yMax * y_scale);

        cur_line_spacing         = static_cast<unsigned>(ceil(ft_font.get_face()->height * y_scale));

        cur_uline_pos            = static_cast<int>(round(ft_font.get_face()->underline_position * y_scale));
        cur_uline_thick          = static_cast<unsigned>(round(ft_font.get_face()->underline_thickness * y_scale));

    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        font_bbox_x = vec2f(0.0f,
                            static_cast<float>(ft_font.get_face()->size->metrics.max_advance >> 6));
        font_bbox_y = vec2f(0.0f,
                            static_cast<float>(ft_font.get_face()->size->metrics.height >> 6));

        cur_line_spacing         = static_cast<int>(font_bbox_y.y);
        cur_uline_pos            = -1;
        cur_uline_thick          = 1;
    }

    font_glyph_bbox_size  = vec2i(static_cast<int>(ceil(font_bbox_x.y) - floor(font_bbox_x.x)),
                                  static_cast<int>(ceil(font_bbox_y.y) - floor(font_bbox_y.x)));

    // allocate texture space
    // glyphs are stacked 16x16 in the texture
    cur_texture._size = vec2ui(font_glyph_bbox_size.x * 16,
                               font_glyph_bbox_size.y * 16);

    // allocate texture destination memory
    try {
        cur_texture._data.reset(new unsigned char[cur_texture._size.x * cur_texture._size.y]);
    }
    catch (std::bad_alloc&) {
        cur_texture._data.reset();
        scm::err() << log::error
                   << "scm::font::face_loader::load_style(): "
                   << "unable to allocate font texture memory of size ('"
                   << cur_texture._size.x * cur_texture._size.y << "byte')"
                   << log::end;

        return (false);
    }

    // set texture type
    // currently only supported is grey (1byte per pixel)
    // mono fonts are also converted to grey
    cur_texture._type = face_loader::gray;

    // clear texture background to black
    memset(cur_texture._data.get(), 0u, cur_texture._size.x * cur_texture._size.y);

    unsigned dst_x;
    unsigned dst_y;

    for (unsigned i = 0; i < 256; ++i) {

        glyph&      cur_glyph = cur_glyph_mapping[i];

        if(FT_Load_Glyph(ft_font.get_face(),
                         FT_Get_Char_Index(ft_font.get_face(), i),
                         FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_LIGHT)) {
            continue;
        }

        if (FT_Render_Glyph(ft_font.get_face()->glyph, FT_RENDER_MODE_LIGHT)) {
            continue;
        }
        FT_Bitmap& bitmap = ft_font.get_face()->glyph->bitmap;

        // calculate the glyphs grid position in the font texture
        dst_x = (i & 0x0F) * font_glyph_bbox_size.x;
        dst_y = cur_texture._size.y - ((i >> 4) + 1) * font_glyph_bbox_size.y;

        vec2i actual_glyph_bbox(bitmap.width, bitmap.rows);

        switch (bitmap.pixel_mode) {
            case FT_PIXEL_MODE_GRAY:
                for (int dy = 0; dy < bitmap.rows; ++dy) {

                    unsigned src_off = dy * bitmap.pitch;
                    unsigned dst_off = dst_x + (dst_y + bitmap.rows - 1 - dy) * cur_texture._size.x;
                    memcpy(cur_texture._data.get() + dst_off, bitmap.buffer + src_off, bitmap.width);
                }
                break;
            case FT_PIXEL_MODE_MONO:
                for (int dy = 0; dy < bitmap.rows; ++dy) {
                    for (int dx = 0; dx < bitmap.pitch; ++dx) {

                        unsigned        src_off     = dx + dy * bitmap.pitch;
                        unsigned char   src_byte    = bitmap.buffer[src_off];

                        for (int bx = 0; bx < 8; ++bx) {

                            unsigned dst_off    = (dst_x + dx * 8 + bx) + (dst_y + bitmap.rows - 1 - dy) * cur_texture._size.x;

                            unsigned char  src_set = src_byte & (0x80 >> bx);
                            unsigned char* plah = &src_byte;

                            cur_texture._data[dst_off] = src_set ? 255u : 0u;
                        }
                    }
                }
                break;
            default:
                continue;
        }

        cur_glyph._tex_lower_left   = vec2i(dst_x, dst_y);
        cur_glyph._tex_upper_right  = cur_glyph._tex_lower_left + actual_glyph_bbox;

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
    }

    // calculate kerning information
    //cur_kerning_table.resize(boost::extents[256][256]);

    //if (ft_font.get_face()->face_flags & FT_FACE_FLAG_KERNING) {
    //    for (unsigned l = 0; l < 256; ++l) {
    //        FT_UInt l_glyph_index = FT_Get_Char_Index(ft_font.get_face(), l);
    //        for (unsigned r = 0; r < 256; ++r) {
    //            FT_UInt     r_glyph_index = FT_Get_Char_Index(ft_font.get_face(), r);
    //            FT_Vector   delta;
    //            FT_Get_Kerning(ft_font.get_face(), l_glyph_index, r_glyph_index,
    //                           FT_KERNING_DEFAULT, &delta);

    //            cur_kerning_table[l][r] = static_cast<char>(delta.x >> 6);
    //        }
    //    }
    //}
    //else {
    //    for (unsigned l = 0; l < 256; ++l) {
    //        for (unsigned r = 0; r < 256; ++r) {
    //            cur_kerning_table[l][r] = 0;
    //        }
    //    }
    //}

    return (true);
}

void face_loader::find_font_styles(const std::string& font_file,
                                   std::map<face::style_type, std::string>& styles) const
{
    typedef std::map<face::style_type, std::string>::value_type val_type;

    using namespace boost::filesystem;

    // insert regular style
    styles.insert(val_type(face::regular, font_file));

    path            font_file_path = path(font_file);
    std::string     font_file_ext  = extension(font_file_path);
    std::string     font_file_base = basename(font_file_path);
    path            font_file_dir  = font_file_path.branch_path();

    // search for italic style
    path    font_file_italic = font_file_dir
                             / (font_file_base + std::string("i") + font_file_ext);
    if (exists(font_file_italic) && !is_directory(font_file_italic)) {
        styles.insert(val_type(face::italic, font_file_italic.string()));
    }

    // search for bold style
    path    font_file_bold = font_file_dir
                           / (font_file_base + std::string("b") + font_file_ext);
    if (exists(font_file_bold) && !is_directory(font_file_bold)) {
        styles.insert(val_type(face::bold, font_file_bold.string()));
    }
    else {
        font_file_bold =  font_file_dir
                        / (font_file_base + std::string("bd") + font_file_ext);
        if (exists(font_file_bold) && !is_directory(font_file_bold)) {
            styles.insert(val_type(face::bold, font_file_bold.string()));
        }
    }

    // search for bold italic style (z or bi name addition)
    path    font_file_bold_italic = font_file_dir
                                  / (font_file_base + std::string("z") + font_file_ext);
    if (exists(font_file_bold_italic) && !is_directory(font_file_bold_italic)) {
        styles.insert(val_type(face::bold_italic, font_file_bold_italic.string()));
    }
    else {
        font_file_bold_italic = font_file_dir
                              / (font_file_base + std::string("bi") + font_file_ext);
        if (exists(font_file_bold_italic) && !is_directory(font_file_bold_italic)) {
            styles.insert(val_type(face::bold_italic, font_file_bold_italic.string()));
        }
    }
}

unsigned face_loader::available_72dpi_size(const std::string& file_name,
                                           unsigned           size,
                                           unsigned           disp_res) const
{
    using namespace boost::filesystem;

    std::string font_file_name;

    if (!check_font_file(file_name, font_file_name)) {
        return (false);
    }

    path        font_file = path(font_file_name);

    unsigned    font_size   = size;
    unsigned    display_res = disp_res;

    detail::ft_library  ft_lib;
    detail::ft_face     ft_font;

    if (!ft_lib.open()) {
        scm::err() << log::error
                   << "scm::font::face_loader::available_72dpi_size(): "
                   << "unable to initialize freetype library"
                   << log::end;
        return (0);
    }

    if (!ft_font.open_face(ft_lib, font_file.file_string())) {
        scm::err() << log::error
                   << "scm::font::face_loader::available_72dpi_size(): "
                   << "unable to load font file ('" << font_file.file_string() << "')"
                   << log::end;
        return (0);
    }

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {

        font_size = (font_size * disp_res + 36) / 72;
    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        std::set<int> available_sizes;

        for (int i = 0; i < ft_font.get_face()->num_fixed_sizes; ++i) {
            available_sizes.insert(ft_font.get_face()->available_sizes[i].height);
        }

        if (available_sizes.empty()) {
            scm::err() << log::error
                       << "scm::font::face_loader::available_72dpi_size(): "
                       << "specified font file ('" << font_file.file_string() << "') "
                       << "contains fixed size font but fails to report available sizes"
                       << log::end;
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
        scm::err() << log::error
                   << "scm::font::face_loader::available_72dpi_size(): "
                   << "specified font file ('" << font_file.file_string() << "') "
                   << "contains unsupported face type"
                   << log::end;
        return (0);
    }

    return (font_size);
}

const face_loader::texture_type& face_loader::get_current_face_texture(face::style_type style) const
{
    face_texture_mapping::const_iterator tex_it = _face_style_textures.find(style);

    if (tex_it == _face_style_textures.end()) {
        tex_it = _face_style_textures.find(face::regular);

        if (tex_it == _face_style_textures.end()) {
            std::stringstream output;

            output << "scm::font::face_loader::get_current_face_texture(): "
                   << "unable to retrieve texture for requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << log::end;

            scm::err() << log::error
                       << output.str();

            throw std::runtime_error(output.str());
        }
    }

    return (tex_it->second);
}

void face_loader::free_texture_resources()
{
    /*face_texture_mapping::iterator tex_it;

    for (tex_it  = _face_style_textures.begin();
         tex_it != ?face_texture_mapping.end();
         ++tex_it) {
        tex_it->second._data.reset();
    }*/

    _face_style_textures.clear();
}

} // namespace font
} // namespace scm
