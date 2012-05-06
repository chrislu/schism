
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TEXTURE_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_H_INCLUDED

#include <scm/gl_core/constants.h>
#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/context_bindable_object.h>
#include <scm/gl_core/render_device/device_resource.h>
#include <scm/gl_core/state_objects/state_objects_fwd.h>
#include <scm/gl_core/texture_objects/texture_objects_fwd.h>

#include <scm/core/math.h>

#include <scm/gl_core/data_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) texture : public context_bindable_object,
                                      public render_device_resource
{
public:
    virtual ~texture();

    uint64          native_handle() const;
    bool            native_handle_resident() const;

protected:
    texture(render_device& in_device);

    void            bind(const render_context& in_context, int in_unit) const;
    void            unbind(const render_context& in_context, int in_unit) const;

    void            bind_image(const render_context& in_context,
                                     unsigned        in_unit,
                                     data_format     in_format,
                                     access_mode     in_access,
                                     int             in_level,
                                     int             in_layer) const;
    void            unbind_image(const render_context& in_context, int in_unit) const;

    bool            make_resident(const render_context&    in_context,
                                  const sampler_state_ptr& in_sstate);
    bool            make_non_resident(const render_context&    in_context);

protected:
    uint64          _native_handle;
    bool            _native_handle_resident;

private:
    friend class render_device;
    friend class render_context;

}; // class texture

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_H_INCLUDED
