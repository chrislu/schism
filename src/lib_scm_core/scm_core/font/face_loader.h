

#ifndef FONT_FACE_LOADER_H_INCLUDED
#define FONT_FACE_LOADER_H_INCLUDED

#include <map>
#include <string>

#include <boost/shared_array.hpp>

#include <scm_core/font/face.h>

#include <scm_core/math/math.h>
#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

class __scm_export face_loader
{
public:
    typedef enum {
        mono        = 0x01,
        gray,
        lcd
    } texture_bitmap_type;

    typedef struct {
        math::vec2ui_t                      _size;
        texture_bitmap_type                 _type;
        boost::shared_array<unsigned char>  _data;
    } texture_type;

protected:
    typedef std::map<face::style_type, texture_type>    face_texture_mapping;

public:
    face_loader(const std::string& /*res_path*/ = std::string(""));
    virtual ~face_loader();

    bool                    load(face&              /*font_face*/,
                                 const std::string& /*file_name*/,
                                 unsigned           /*size*/     = 12,
                                 unsigned           /*disp_res*/ = 72);

    const texture_type&     get_current_face_texture(face::style_type /*style*/) const;
    void                    free_texture_resources();

protected:
    std::string             _resources_path;
    face_texture_mapping    _face_style_textures;

    void                    find_font_styles(const std::string&                       /*font_file*/,
                                             std::map<face::style_type, std::string>& /*styles*/) const;
    bool                    load_style(face::style_type   /*style*/,
                                       const std::string& /*file_name*/,
                                       face&              /*font_face*/,
                                       unsigned           /*size*/,
                                       unsigned           /*disp_res*/);
}; // class face_loader

} // namespace font
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // FONT_FACE_LOADER_H_INCLUDED
