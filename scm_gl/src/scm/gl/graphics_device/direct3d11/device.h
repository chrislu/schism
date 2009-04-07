
#ifndef SCM_GL_DIRECT3D11_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_DIRECT3D11_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/gl/graphics_device/device.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) direct3d_device : public device
{
public:
    virtual ~direct3d_device();

protected:
    direct3d_device(const device_initializer& init,
                    const device_context_config& cfg);

    virtual bool                setup_render_context(const device_context_config& cfg,
                                                     unsigned                    feature_level);
private:
    friend device_ptr create_device(const device_initializer&, const device_context_config&);

}; // class device

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_DIRECT3D11_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED
