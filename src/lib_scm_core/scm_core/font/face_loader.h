

#ifndef FONT_FACE_LOADER_H_INCLUDED
#define FONT_FACE_LOADER_H_INCLUDED

#include <string>

#include <scm_core/platform/platform.h>

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
        unsigned                    _width;
        unsigned                    _height;
        texture_bitmap_type         _type;
        unsigned char*              _data;
    } face_style_texture;

public:
    face_loader(const std::string& /*res_path*/ = std::string(""));
    virtual ~face_loader();

    bool            load(const std::string& /*file_name*/,
                         unsigned           /*size*/     = 12,
                         unsigned           /*disp_res*/ = 72);

protected:
    std::string     _resources_path;

}; // class face_loader

} // namespace font
} // namespace scm

#endif // FONT_FACE_LOADER_H_INCLUDED
