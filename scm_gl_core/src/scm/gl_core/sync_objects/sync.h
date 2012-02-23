
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_SYNC_H_INCLUDED
#define SCM_GL_CORE_SYNC_H_INCLUDED

#include <scm/core/numeric_types.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/device_resource.h>
#include <scm/gl_core/texture_objects/texture_objects_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

struct __GLsync;

namespace scm {
namespace gl {

class __scm_export(gl_core) sync : public render_device_child
{
protected:
    typedef struct __GLsync* GLsync;

public:
    virtual ~sync();

protected:
    sync(render_device& in_device);

    sync_wait_result            client_wait(const render_context& in_context,
                                                  scm::uint64     in_timeout = sync_timeout_ignored,
                                                  bool            in_flush   = true) const;
    void                        server_wait(const render_context& in_context,
                                                  scm::uint64     in_timeout = sync_timeout_ignored) const;
    sync_status                 status(const render_context& in_context) const;

    GLsync                      object() const;
    void                        delete_sync();

protected:
    GLsync                      _gl_sync_object;

private:
    friend class render_device;
    friend class render_context;

}; // class sync

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_SYNC_H_INCLUDED
