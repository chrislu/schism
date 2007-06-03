

#ifndef FONT_FACE_LOADER_H_INCLUDED
#define FONT_FACE_LOADER_H_INCLUDED

#include <string>

#include <scm_core/platform/platform.h>

namespace scm {
namespace font {

class __scm_export face_loader
{
public:
    face_loader();
    face_loader(const std::string& /*font_dir*/);
    virtual ~face_loader();

    bool        load(const std::string& /*file_name*/,
                     unsigned           /*size*/     = 12,
                     unsigned           /*disp_res*/ = 72);

protected:


    std::string     _font_directory;

}; // class face_loader

} // namespace font
} // namespace scm

#endif // FONT_FACE_LOADER_H_INCLUDED
