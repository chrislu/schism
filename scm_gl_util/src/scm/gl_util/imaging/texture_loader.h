
#ifndef SCM_GL_UTIL_TEXTURE_LOADER_H_INCLUDED
#define SCM_GL_UTIL_TEXTURE_LOADER_H_INCLUDED

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/texture_objects/texture_objects_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) texture_loader
{
public:

    texture_2d_ptr      load_texture_2d(render_device&       in_device,
                                        const std::string&   in_image_path,
                                        bool                 in_create_mips,
                                        bool                 in_color_mips  = false,
                                        const data_format    in_force_internal_format = FORMAT_NULL);

protected:

private:
}; // class texture_loader

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXTURE_LOADER_H_INCLUDED
