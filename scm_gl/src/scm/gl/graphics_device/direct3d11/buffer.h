
#ifndef SCM_GL_DIRECT3D11_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_DIRECT3D11_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/gl/graphics_device/buffer.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class direct3d_device;

class __scm_export(ogl) direct3d_buffer : public buffer
{
public:
    virtual ~direct3d_buffer();

protected:
    direct3d_buffer(direct3d_device& dev);

protected:

    friend class scm::gl::direct3d_device;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_DIRECT3D11_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED
