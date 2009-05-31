
#ifndef SCM_GL_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED

#include <vector>

#include <scm/core/math.h>

#include <scm/gl/graphics_device/device_resource.h>
#include <scm/gl/graphics_device/formats.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class device_context_descriptor;

class __scm_export(ogl) device_context : public device_resource
{
public:


public:
    virtual ~device_context();

protected:
    device_context(device& dev);

private:
    

}; // class device_context

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_DEVICE_CONTEXT_H_INCLUDED_DEVICE_H_INCLUDED
