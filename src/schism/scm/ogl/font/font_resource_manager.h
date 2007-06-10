
#ifndef GL_FONT_RESOURCE_MANAGER_H_INCLUDED
#define GL_FONT_RESOURCE_MANAGER_H_INCLUDED

#include <cstddef>
#include <string>

#include <scm/ogl/font/font.h>
#include <scm/core/resource/resource_manager.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(ogl) font_resource_manager : public res::resource_manager<font_face_resource>
{
public:
    font_resource_manager();
    virtual ~font_resource_manager();

    bool        initialize();
    bool        shutdown();

}; // class font_resource_manager

} // namespace gl
} // namespace scm

#endif // GL_FONT_RESOURCE_MANAGER_H_INCLUDED
