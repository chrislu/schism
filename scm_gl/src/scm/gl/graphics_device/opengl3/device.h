
#ifndef SCM_GL_OPENGL3_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_OPENGL3_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/gl/graphics_device/device.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

namespace detail {
} // namespace detail

struct device_initializer;

class __scm_export(ogl) opengl_device : public device
{
public:
    enum feature_levels {
        OPENGL_FEATURE_LEVEL_2_1    = 2010u,
        OPENGL_FEATURE_LEVEL_3_0    = 3000u,
        OPENGL_FEATURE_LEVEL_3_1    = 3010u
    };

public:
    virtual ~opengl_device();

protected:
    opengl_device(const device_initializer& init,
                  const device_context_config& cfg);

    virtual bool                setup_render_context(const device_context_config& cfg,
                                                     unsigned                    feature_level) = 0;

protected:
private:
    friend device_ptr create_device(const device_initializer&, const device_context_config&);
}; // class device

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_OPENGL3_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED
