

#ifndef FONT_FACE_LOADER_H_INCLUDED
#define FONT_FACE_LOADER_H_INCLUDED

#include <map>
#include <string>

#include <boost/shared_array.hpp>

#include <scm/core/font/face.h>

#include <scm/core/math/math.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

class __scm_export(core) face_loader
{
public:
    typedef enum {
        mono        = 0x01,
        gray,
        lcd
    } texture_bitmap_type;

    typedef struct {
        scm::math::vec2ui                   _size;
        texture_bitmap_type                 _type;
        boost::shared_array<unsigned char>  _data;
    } texture_type;

protected:
    typedef std::map<face::style_type, texture_type>    face_texture_mapping;

public:
    face_loader();
    virtual ~face_loader();

    bool                    load(face&              /*font_face*/,
                                 const std::string& /*file_name*/,
                                 unsigned           /*size*/     = 12,
                                 unsigned           /*disp_res*/ = 72);

    void                    set_font_resource_path(const std::string& /*res_path*/);

    const texture_type&     get_current_face_texture(face::style_type /*style*/) const;
    void                    free_texture_resources();

protected:
    std::string             _resources_path;
    face_texture_mapping    _face_style_textures;

    void                    find_font_styles(const std::string&                       /*font_file*/,
                                             std::map<face::style_type, std::string>& /*styles*/) const;
    bool                    check_font_file(const std::string& /*in_file_name*/,
                                            std::string&       /*out_file_path*/) const;
    unsigned                available_72dpi_size(const std::string& /*file_name*/,
                                                 unsigned           /*size*/,
                                                 unsigned           /*disp_res*/) const;
    bool                    load_style(face::style_type   /*style*/,
                                       const std::string& /*file_name*/,
                                       face&              /*font_face*/,
                                       unsigned           /*size*/,
                                       unsigned           /*disp_res*/);
}; // class face_loader

} // namespace font
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // FONT_FACE_LOADER_H_INCLUDED
