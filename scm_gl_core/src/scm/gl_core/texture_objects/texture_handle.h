
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TEXTURE_RESIDENT_HANDLE_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_RESIDENT_HANDLE_H_INCLUDED

#include <iosfwd>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/device_resource.h>
#include <scm/gl_core/state_objects/state_objects_fwd.h>
#include <scm/gl_core/texture_objects/texture_objects_fwd.h>

#include <scm/core/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) texture_handle : public render_device_resource
{
public:
    virtual ~texture_handle();

    uint64          native_handle() const;
    bool            native_handle_resident() const;

    void            print(std::ostream& os) const {};

protected:
    texture_handle(      render_device& in_device,
                   const texture&       in_texture,
                   const sampler_state& in_sampler);

    bool            make_resident(      render_device& in_device,
                                  const texture&       in_texture,
                                  const sampler_state& in_sampler,
                                        std::ostream&  out_stream);
    bool            make_non_resident(const render_device& in_device,
                                            std::ostream&  out_stream);

protected:
    uint64          _native_handle;
    bool            _native_handle_resident;

private:
    friend class render_device;
    friend class render_context;

}; // class texture_handle

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_RESIDENT_HANDLE_H_INCLUDED
