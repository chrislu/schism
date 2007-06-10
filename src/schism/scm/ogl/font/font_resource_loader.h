
#ifndef GL_FONT_RESOURCE_LOADER_H_INCLUDED
#define GL_FONT_RESOURCE_LOADER_H_INCLUDED

#include <scm/core/font/face_loader.h>

#include <scm/ogl/font/font.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl
{

class __scm_export(ogl) font_resource_loader : protected font::face_loader
{
public:
    font_resource_loader();
    virtual ~font_resource_loader();

    using font::face_loader::set_font_resource_path;

    virtual font_face       load(const std::string& /*file_name*/,
                                 unsigned           /*size*/     = 12,
                                 unsigned           /*disp_res*/ = 72);

}; // class font_resource_loader

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //GL_FONT_RESOURCE_LOADER_H_INCLUDED
