
#include "font_resource_loader.h"

#include <vector>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/console.h>

#include <scm/ogl.h>
#include <scm/ogl/utilities/error_checker.h>

using namespace scm;
using namespace scm::gl;

font_resource_loader::font_resource_loader()
{
}

font_resource_loader::~font_resource_loader()
{
}

font_face font_resource_loader::load(const std::string& file_name,
                                     unsigned           size,
                                     unsigned           disp_res)
{
    font_face       ret_val = font_face();

    // check file
    using namespace boost::filesystem;

    std::string font_file_name;

    if (!check_font_file(file_name, font_file_name)) {
        return (ret_val);
    }

    path        font_file = path(font_file_name);

    // check for pixel size
    unsigned    font_size   = size;
    unsigned    display_res = disp_res;

    font_size = available_72dpi_size(font_file.file_string(),
                                     size,
                                     disp_res);

    if (font_size == 0) {
        return (ret_val);
    }

    // build descriptor from this and ask resource manager\
    // if this is allready here
    font_descriptor     desc;

    desc._name  = font_file.file_string();
    desc._size  = font_size;

    if (ogl.get().get_font_manager().is_loaded(desc)) {
        ret_val = ogl.get().get_font_manager().retrieve_instance(desc);

        return (ret_val);
    }

    // ok we have a new instance at our hands
    ret_val = ogl.get().get_font_manager().create_instance(desc);

    if (!font::face_loader::load(ret_val.get(),
                                file_name,
                                size,
                                display_res)) {
        return (ret_val);
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

        if (ret_val.get().has_style(*style_it)) {

            ret_val.get()._style_textures[*style_it].reset(new texture_2d_rect);
            
            texture_2d_rect& cur_tex = *ret_val.get()._style_textures[*style_it];

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
                                   << "font_resource_loader::load(): "
                                   << "error uploading face texture: "
                                   << _error_check.get_error_string()
                                   << std::endl;

                ret_val.get().cleanup_textures();
                free_texture_resources();
                return (font_face());
            }

            cur_tex.tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            cur_tex.tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            cur_tex.tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            cur_tex.tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }

    return (ret_val);
}
