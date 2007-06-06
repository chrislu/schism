
#include "face_loader.h"

#include <cassert>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>

#include <scm_core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm_core/utilities/boost_warning_enable.h>

#include <scm_core/console.h>
#include <scm_core/exception/system_exception.h>

#include <scm_core/font/face.h>
#include <scm_core/font/detail/freetype_types.h>

using namespace scm::font;

face_loader::face_loader(const std::string& res_path)
  : _resources_path(res_path)
{
}

face_loader::~face_loader()
{
    free_texture_resources();
}

bool face_loader::load(face&              font_face,
                       const std::string& file_name,
                       unsigned           size,
                       unsigned           disp_res)
{
    free_texture_resources();
    font_face.clear();

    using namespace boost::filesystem;

    path        font_file   = path(file_name);

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

    // find font styles ({name}i|b|z|bi.{ext})
    typedef std::map<face::style_type, std::string> styles_container;

    styles_container styles;
    find_font_styles(font_file.file_string(), styles);

    // load font styles
    styles_container::const_iterator style_it;

    for (style_it  = styles.begin();
         style_it != styles.end();
         ++style_it) {

        if (!load_style(style_it->first,
                        style_it->second,
                        font_face,
                        size,
                        disp_res)) {

            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "face_loader::load(): "
                               << "error loading face style (id ='" << style_it->first << "') "
                               << "using font file ('" << style_it->second << "')"
                               << std::endl;

            free_texture_resources();
            font_face.clear();

            return (false);
        }
    }

    font_face._name             = font_file.file_string();

    return (true);
}

bool face_loader::load_style(face::style_type   style,
                             const std::string& file_name,
                             face&              font_face,
                             unsigned           size,
                             unsigned           disp_res)
{
    unsigned    font_size   = size;
    unsigned    display_res = disp_res;

    face::kerning_table&            cur_kerning_table = font_face._glyph_mappings[style]._kerning_table;
    face::character_glyph_mapping&  cur_glyph_mapping = font_face._glyph_mappings[style]._glyph_mapping;
    texture_type&                   cur_texture       = _face_style_textures[style];

    detail::ft_library  ft_lib;
    detail::ft_face     ft_font;

    if (!ft_lib.open()) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load_style(): "
                           << "unable to initialize freetype library"
                           << std::endl;

        return (false);
    }

    if (!ft_font.open_face(ft_lib, file_name)) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load_style(): "
                           << "unable to load font file ('" << file_name << "')"
                           << std::endl;

        return (false);
    }

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_SCALABLE) {

        font_face._size_at_72dpi = (font_size * disp_res + 36) / 72;
    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        std::set<int> available_sizes;

        for (int i = 0; i < ft_font.get_face()->num_fixed_sizes; ++i) {
            available_sizes.insert(ft_font.get_face()->available_sizes[i].height);
        }

        if (available_sizes.empty()) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "face_loader::load_style(): "
                               << "specified font file ('" << file_name << "') "
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

        font_face._size_at_72dpi = font_size;
        // ok bitmap fonts are in pixel sizes (i.e. 72dpi)
        // so scale the pixel size to our display resolution
        // for the following functions

        font_size = (72 * font_size + disp_res / 2) / disp_res;
    }
    else {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load_style(): "
                           << "specified font file ('" << file_name << "') "
                           << "contains unsupported face type"
                           << std::endl;

        return (false);
    }

    if (FT_Set_Char_Size(ft_font.get_face(), 0, font_size << 6, 0, disp_res) != 0) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load_style(): "
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

        font_face._line_spacing  = static_cast<int>(math::ceil(ft_font.get_face()->height * y_scale));

    }
    else if (ft_font.get_face()->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        font_bbox_x = math::vec2f_t(0.0f,
                                    static_cast<float>(ft_font.get_face()->size->metrics.max_advance >> 6));
        font_bbox_y = math::vec2f_t(0.0f,
                                    static_cast<float>(ft_font.get_face()->size->metrics.height >> 6));

        font_face._line_spacing  = static_cast<int>(font_bbox_y.y);
    }

    font_glyph_bbox_size  = math::vec2i_t(static_cast<int>(math::ceil(font_bbox_x.y) - math::floor(font_bbox_x.x)),
                                          static_cast<int>(math::ceil(font_bbox_y.y) - math::floor(font_bbox_y.x)));

    // allocate texture space
    // glyphs are stacked 16x16 in the texture
    cur_texture._size = math::vec2ui_t(font_glyph_bbox_size.x * 16,
                                       font_glyph_bbox_size.y * 16);

    // allocate texture destination memory
    try {
        cur_texture._data.reset(new unsigned char[cur_texture._size.x * cur_texture._size.y]);
    }
    catch (std::bad_alloc&) {
        cur_texture._data.reset();
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "face_loader::load_style(): "
                           << "unable to allocate font texture memory of size ('"
                           << cur_texture._size.x * cur_texture._size.y << "byte')"
                           << std::endl;

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

        math::vec2i_t actual_glyph_bbox(bitmap.width, bitmap.rows);

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

        cur_glyph._tex_lower_left   = math::vec2i_t(dst_x, dst_y);
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
        cur_glyph._bearing          = math::vec2i_t(ft_font.get_face()->glyph->metrics.horiBearingX >> 6,
                                                      (ft_font.get_face()->glyph->metrics.horiBearingY >> 6)
                                                    - (ft_font.get_face()->glyph->metrics.height >> 6));
    }

    // calculate kerning information
    cur_kerning_table.resize(boost::extents[256][256]);

    if (ft_font.get_face()->face_flags & FT_FACE_FLAG_KERNING) {
        for (unsigned l = 0; l < 256; ++l) {
            FT_UInt l_glyph_index = FT_Get_Char_Index(ft_font.get_face(), l);
            for (unsigned r = 0; r < 256; ++r) {
                FT_UInt     r_glyph_index = FT_Get_Char_Index(ft_font.get_face(), r);
                FT_Vector   delta;
                FT_Get_Kerning(ft_font.get_face(), l_glyph_index, r_glyph_index,
                               FT_KERNING_DEFAULT, &delta);

                cur_kerning_table[l][r] = static_cast<char>(delta.x >> 6);
            }
        }
    }
    else {
        for (unsigned l = 0; l < 256; ++l) {
            for (unsigned r = 0; r < 256; ++r) {
                cur_kerning_table[l][r] = 0;
            }
        }
    }

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
    path            font_file_italic =  font_file_dir
                                      / (font_file_base + std::string("i") + font_file_ext);
    if (exists(font_file_italic) && !is_directory(font_file_italic)) {
        styles.insert(val_type(face::italic, font_file_italic.string()));
    }

    // search for bold style
    path            font_file_bold =    font_file_dir
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
    path            font_file_bold_italic = font_file_dir
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

const face_loader::texture_type& face_loader::get_current_face_texture(face::style_type style) const
{
    face_texture_mapping::const_iterator tex_it = _face_style_textures.find(style);

    if (tex_it == _face_style_textures.end()) {
        tex_it = _face_style_textures.find(face::regular);

        if (tex_it == _face_style_textures.end()) {
            std::stringstream output;

            output << "face_loader::get_current_face_texture(): "
                   << "unable to retrieve texture for requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << std::endl;

            console.get() << con::log_level(con::error)
                          << output.str();

            throw scm::core::system_exception(output.str());
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
