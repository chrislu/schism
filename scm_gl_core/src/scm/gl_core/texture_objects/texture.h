
#ifndef SCM_GL_CORE_TEXTURE_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_H_INCLUDED

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/frame_buffer_objects/render_target.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/device_resource.h>

#include <scm/core/math.h>

#include <scm/gl_core/data_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

namespace util {

unsigned        max_mip_levels(const unsigned      in_tex_size);
unsigned        max_mip_levels(const math::vec2ui& in_tex_size);
unsigned        max_mip_levels(const math::vec3ui& in_tex_size);
unsigned        mip_level_dimensions(const unsigned      in_tex_size, unsigned in_level);
math::vec2ui    mip_level_dimensions(const math::vec2ui& in_tex_size, unsigned in_level);
math::vec3ui    mip_level_dimensions(const math::vec3ui& in_tex_size, unsigned in_level);

} // namespace util

class __scm_export(gl_core) texture : public render_target
{
public:
    virtual ~texture();

protected:
    texture(render_device& in_device);

    void            bind(const render_context& in_context, int in_unit) const;
    void            unbind(const render_context& in_context, int in_unit) const;

    unsigned        texture_binding() const;

    virtual bool    image_sub_data(const render_context& in_context,
                                   const texture_region& in_region,
                                   const unsigned        in_level,
                                   const data_format     in_data_format,
                                   const void*const      in_data) = 0;

protected:
    unsigned        _gl_texture_binding;

private:

    friend class render_device;
    friend class render_context;
}; // class texture

} // namespace gl
} // namespace scm

#include "texture.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_H_INCLUDED
