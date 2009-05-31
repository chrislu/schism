
#ifndef SCM_GL_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED

#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>

#include <scm/core/pointer_types.h>

#include <scm/gl/graphics_device/scmgl.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class device_resource;
class device_context;

typedef scm::shared_ptr<device_resource>    device_resource_ptr;
typedef scm::shared_ptr<device_context>     device_context_ptr;

class __scm_export(ogl) device : boost::noncopyable
{
private:
    //typedef scm::weak_ptr<device_resource>  device_resource_weak_ptr;
    //typedef boost::unordered_map<device_resource_weak_ptr, 

public:
    virtual ~device();

    unsigned                    feature_level() const;
    device_type                 type() const;


    //virtual bool                bind_context_to_current_thread(const device_context_ptr& /*ctx*/) const = 0;
    //virtual bool                swap_buffers(const device_context_ptr& /*ctx*/) const = 0;


protected:
    device();

protected:
    unsigned                    _feature_level;
    device_type                 _type;

private:


}; // class device

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_DEVICE_H_INCLUDED_DEVICE_H_INCLUDED
