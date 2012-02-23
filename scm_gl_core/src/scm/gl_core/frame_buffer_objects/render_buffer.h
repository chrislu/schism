
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_RENDER_BUFFER_H_INCLUDED
#define SCM_GL_CORE_RENDER_BUFFER_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/frame_buffer_objects/render_target.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/context_bindable_object.h>
#include <scm/gl_core/render_device/device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) render_buffer_desc
{
    render_buffer_desc(const math::vec2ui& in_size,
                       const data_format   in_format,
                       const unsigned      in_samples = 1);

    bool operator==(const render_buffer_desc& rhs) const;
    bool operator!=(const render_buffer_desc& rhs) const;

    math::vec2ui    _size;
    data_format     _format;
    unsigned        _samples;
}; // struct render_buffer_desc

class __scm_export(gl_core) render_buffer : public context_bindable_object,
                                            public render_target,
                                            public render_device_resource
{
public:
    virtual ~render_buffer();

    const render_buffer_desc& descriptor() const;
    void                      print(std::ostream& os) const {};

    data_format               format() const;
    math::vec2ui              dimensions() const;
    unsigned                  array_layers() const;
    unsigned                  mip_map_layers() const;
    unsigned                  samples() const;

protected:
    render_buffer(render_device&            in_device,
                  const render_buffer_desc& in_desc);

    unsigned                  object_id() const;
    unsigned                  object_target() const;

protected:
    render_buffer_desc      _descriptor;

private:
    friend class render_device;
    friend class render_context;

}; // class render_buffer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_RENDER_BUFFER_H_INCLUDED
