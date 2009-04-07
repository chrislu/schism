
#ifndef SCM_GL_DIRECT3D11_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_DIRECT3D11_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/gl/graphics_device/device_context.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class direct3d_device;

class __scm_export(ogl) direct3d_device_context : public device_context
{
public:
    virtual ~direct3d_device_context();

protected:
    direct3d_device_context(direct3d_device& dev);

private:
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_DIRECT3D11_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED
