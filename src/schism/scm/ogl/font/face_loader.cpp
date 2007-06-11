
#include "face_loader.h"

#include <vector>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/console.h>

#include <scm/ogl/gl.h>
#include <scm/ogl/utilities/error_checker.h>

using namespace scm;
using namespace scm::gl;

face_loader::face_loader()
{
}

face_loader::~face_loader()
{
}

scm::gl::face_ptr face_loader::load(const std::string& file_name,
                                    unsigned           size,
                                    unsigned           disp_res)
{
    face_ptr  font_face = face_ptr(new scm::gl::face);

    if (!load(*font_face,
              file_name,
              size,
              disp_res)) {

        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "scm::gl::face_loader::load(): "
                           << "error loading font face: "
                           << file_name
                           << std::endl;

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

            cur_tex.bind();
            cur_tex.tex_image(0,
                              GL_ALPHA,
                              get_current_face_texture(*style_it)._size.x,
                              get_current_face_texture(*style_it)._size.y,
                              GL_ALPHA,
                              GL_UNSIGNED_BYTE,
                              (void*)get_current_face_texture(*style_it)._data.get());

            if (!_error_check.ok()) {
                scm::console.get() << scm::con::log_level(scm::con::error)
                                   << "scm::gl::face_loader::load(): "
                                   << "error uploading face texture: "
                                   << _error_check.get_error_string()
                                   << std::endl;

                font_face->cleanup_textures();
                free_texture_resources();
                font_face.reset();
                return (font_face);
            }

            cur_tex.tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            cur_tex.tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            cur_tex.tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            cur_tex.tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }

    return (font_face);
}
