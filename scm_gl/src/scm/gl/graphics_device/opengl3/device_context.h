
#ifndef SCM_GL_OPENGL3_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_OPENGL3_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/gl/graphics_device/device_context.h>

#include <scm/gl/graphics_device/scmgl.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class opengl_device;

class __scm_export(ogl) opengl_device_context : public device_context
{
public:

public:
    virtual ~opengl_device_context();

    const handle            context_handle() const;

protected:
    opengl_device_context(opengl_device& dev);

protected:
    handle                  _context_handle;

};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_OPENGL3_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED
