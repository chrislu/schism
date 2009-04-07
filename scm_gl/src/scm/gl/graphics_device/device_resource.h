
#ifndef SCM_GL_DEVICE_RESOURCE_DEVICE_RESOURCE_H_INCLUDED
#define SCM_GL_DEVICE_RESOURCE_DEVICE_RESOURCE_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class device;

class __scm_export(ogl) device_resource : boost::noncopyable
{
public:
    virtual ~device_resource();

    const device&       owning_device() const;

protected:
    device_resource(device& owning_device);

private:
    device&             _owning_device;

}; // class device_resource

} // namespace scm
} // namespace gl

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_DEVICE_RESOURCE_DEVICE_RESOURCE_H_INCLUDED
