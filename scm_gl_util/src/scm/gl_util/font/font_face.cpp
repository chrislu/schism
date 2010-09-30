
#include "font_face.h"

#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/filesystem.hpp>
//#include <boost/tuple/tuple.hpp>

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
find_font_style_files(const std::string&         in_regular_font_file,
                      std::vector<std::string>&  out_font_style_files)
{
    using namespace boost::filesystem;

    out_font_style_files.clear();
    out_font_style_files.resize(font_face::style_count);

    // insert regular style
    out_font_style_files[font_face::style_regular] = in_regular_font_file;

    path            font_file_path = path(in_regular_font_file);
    std::string     font_file_ext  = extension(font_file_path);
    std::string     font_file_base = basename(font_file_path);
    path            font_file_dir  = font_file_path.branch_path();

    // search for italic style
    path    font_file_italic = font_file_dir
                             / (font_file_base + std::string("i") + font_file_ext);
    if (exists(font_file_italic) && !is_directory(font_file_italic)) {
        out_font_style_files[font_face::style_italic] = font_file_italic.string();
    }

    // search for bold style
    path    font_file_bold = font_file_dir
                           / (font_file_base + std::string("b") + font_file_ext);
    if (exists(font_file_bold) && !is_directory(font_file_bold)) {
        out_font_style_files[font_face::style_bold] = font_file_bold.string();
    }
    else {
        font_file_bold =  font_file_dir
                        / (font_file_base + std::string("bd") + font_file_ext);
        if (exists(font_file_bold) && !is_directory(font_file_bold)) {
            out_font_style_files[font_face::style_bold] = font_file_bold.string();
        }
    }

    // search for bold italic style (z or bi name addition)
    path    font_file_bold_italic = font_file_dir
                                  / (font_file_base + std::string("z") + font_file_ext);
    if (exists(font_file_bold_italic) && !is_directory(font_file_bold_italic)) {
        out_font_style_files[font_face::style_bold_italic] = font_file_bold_italic.string();
    }
    else {
        font_file_bold_italic = font_file_dir
                              / (font_file_base + std::string("bi") + font_file_ext);
        if (exists(font_file_bold_italic) && !is_directory(font_file_bold_italic)) {
            out_font_style_files[font_face::style_bold_italic] = font_file_bold_italic.string();
        }
    }
}
#if 0
math::vec2ui
styles_glyph_bbox(const std::vector<std::string>& font_style_files)
{
    using namespace scm::math;

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

}
#endif
} // namesapce detail

font_face::font_face(const render_device_ptr& device,
                     const std::string&       font_file,
                     unsigned                 point_size,
                     unsigned                 display_dpi)
  : _font_styles(style_count)
  , _font_styles_available(style_count)
  , _size_at_72dpi(0)
{
    try {
        if (!detail::check_file(font_file)) {
            std::ostringstream s;
            s << "font_face::font_face(): "
                << "font file missing or is a directory ('" << font_file << "')";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }
        std::vector<std::string>    font_style_files;
        detail::find_font_style_files(font_file, font_style_files);
        for (int i = 0; i < style_count; ++i) {
            if (!font_style_files[i].empty()) {
                _font_styles_available[i] = true;
            }
            else {
                _font_styles_available[i] = false;
            }
        }
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
font_face::size_at_72dpi() const
{
    return (_size_at_72dpi);
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
font_face::line_spacing(style_type s) const
{
    return (0);
}

int
font_face::kerning(char l, char r, style_type s) const
{
    return (0);
}

int
font_face::underline_position(style_type s) const
{
    return (0);
}

int
font_face::underline_thickness(style_type s) const
{
    return (0);
}

void
font_face::cleanup()
{
    _font_styles.clear();
    _font_styles_available.clear();
    _font_styles_texture_array.reset();
}

} // namespace gl
} // namespace scm



















#if 0
#include <stdexcept>

#include <scm/core.h>

namespace scm {
namespace gl_classic {

// gl_font_face
face::face()
{
}

face::~face()
{
    cleanup_textures();
}

const texture_2d_rect& face::get_glyph_texture(font::face::style_type style) const
{
    style_textur_container::const_iterator style_it = _style_textures.find(style);

    if (style_it == _style_textures.end()) {
        style_it = _style_textures.find(font::face::regular);

        if (style_it == _style_textures.end()) {
            std::stringstream output;

            output << "scm::gl_classic::face::get_glyph(): "
                   << "unable to retrieve requested style (id = '" << style << "') "
                   << "fallback to regular style failed!" << std::endl;

            scm::err() << log::error
                       << output.str();

            throw std::runtime_error(output.str());
        }
    }

    return (*style_it->second.get());
}

void face::cleanup_textures()
{
    //style_textur_container::iterator style_it;

    //for (style_it  = _style_textures.begin();
    //     style_it != _style_textures.end();
    //     ++style_it) {
    //    style_it->second.reset();
    //}
    _style_textures.clear();
}

} // namespace gl_classic
} // namespace scm
#endif
