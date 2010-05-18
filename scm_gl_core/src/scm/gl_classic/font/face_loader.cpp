
#include "face_loader.h"

#include <vector>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/log.h>

#include <scm/gl_classic/opengl.h>
#include <scm/gl_classic/utilities/error_checker.h>

namespace scm {
namespace gl_classic {

face_loader::face_loader()
{
}

face_loader::~face_loader()
{
}

scm::gl_classic::face_ptr face_loader::load(const std::string& file_name,
                                    unsigned           size,
                                    unsigned           disp_res)
{
    face_ptr  font_face = face_ptr(new scm::gl_classic::face);

    if (!load(*font_face,
              file_name,
              size,
              disp_res)) {

        scm::err() << log::error
                   << "scm::gl_classic::face_loader::load(): "
                   << "error loading font face: "
                   << file_name
                   << log::end;

        font_face->cleanup_textures();
        free_texture_resources();
        font_face.reset();
        return (font_face);
    }

    typedef std::vector<font::face::style_type> style_vec;
    
    style_vec   styles;
    styles.push_back(font::face::regular);
    styles.push_back(font::face::italic);
    styles.push_back(font::face::bold);
    styles.push_back(font::face::bold_italic);

    error_checker _error_check;

    for (style_vec::const_iterator style_it = styles.begin();
         style_it != styles.end();
         ++style_it) {

        if (font_face->has_style(*style_it)) {

            font_face->_style_textures[*style_it].reset(new texture_2d_rect);
            
            texture_2d_rect& cur_tex = *font_face->_style_textures[*style_it];

            //cur_tex.bind();
            cur_tex.image_data(0,
                               GL_ALPHA8,
                               get_current_face_texture(*style_it)._size.x,
                               get_current_face_texture(*style_it)._size.y,
                               GL_ALPHA,
                               GL_UNSIGNED_BYTE,
                               (void*)get_current_face_texture(*style_it)._data.get());

            if (!_error_check.ok()) {
                scm::err() << log::error
                           << "scm::gl_classic::face_loader::load(): "
                           << "error uploading face texture: "
                           << _error_check.error_string()
                           << log::end;

                font_face->cleanup_textures();
                free_texture_resources();
                font_face.reset();
                return (font_face);
            }

            cur_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            cur_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            cur_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            cur_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }

    return (font_face);
}

} // namespace gl_classic
} // namespace scm
