
#ifndef SCM_GL_CORE_SYNC_H_INCLUDED
#define SCM_GL_CORE_SYNC_H_INCLUDED

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

    //client|server_wait(in_context);
    //sync_state()

    GLsync                      gl_object() const;
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
