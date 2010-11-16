
#ifndef SCM_GL_CORE_TEXTURE_BUFFER_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_BUFFER_H_INCLUDED

#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/texture_objects/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

#if 0
struct __scm_export(gl_core) texture_buffer_desc
{
    texture_buffer_desc(const data_format   in_format);

    bool operator==(const texture_buffer_desc& rhs) const;
    bool operator!=(const texture_buffer_desc& rhs) const;

    data_format     _format;
}; // struct texture_buffer_desc

class __scm_export(gl_core) texture_buffer : public texture
{
public:
    virtual ~texture_buffer();

    const texture_buffer_desc&  descriptor() const;
    void                        print(std::ostream& os) const {};

    data_format                 format() const;

protected:
    void                        bind(const render_context& in_context, int in_unit) const;
    void                        unbind(const render_context& in_context, int in_unit) const;

    unsigned                    texture_binding() const;

    void            generate_mipmaps(const render_context& in_context);
    virtual bool    image_sub_data(const render_context& in_context,
                                   const texture_region& in_region,
                                   const unsigned        in_level,
                                   const data_format     in_data_format,
                                   const void*const      in_data) = 0;

protected:
    unsigned        _gl_texture_binding;

protected:
    texture_buffer(render_device&             in_device,
                   const texture_buffer_desc& in_desc);

    bool                        image_sub_data(const render_context& in_context,
                                               const texture_region& in_region,
                                               const unsigned        in_level,
                                               const data_format     in_data_format,
                                               const void*const      in_data);

protected:
    texture_buffer_desc         _descriptor;

private:

    friend class render_device;
    friend class render_context;
}; // class texture_1d
#endif
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_BUFFER_H_INCLUDED
