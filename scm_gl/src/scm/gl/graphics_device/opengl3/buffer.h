
#ifndef SCM_GL_OPENGL3_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_OPENGL3_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/gl/graphics_device/buffer.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class opengl_device;

class __scm_export(ogl) opengl_buffer : public buffer
{
public:
    virtual ~opengl_buffer();

protected:
    opengl_buffer(opengl_device& dev);

protected:

    friend class scm::gl::opengl_device;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_OPENGL3_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED
