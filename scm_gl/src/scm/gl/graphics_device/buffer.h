
#ifndef SCM_GL_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/gl/graphics_device/device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class device;

enum buffer_type {
    BUFFER_TYPE_VERTEX_BUFFER       = 0x01l,
    BUFFER_TYPE_INDEX_BUFFER        = 0x02l,
    BUFFER_TYPE_PIXEL_BUFFER        = 0x04l,
    BUFFER_TYPE_UNIFORM_BUFFER      = 0x08l,
    BUFFER_TYPE_STREAM_OUT_BUFFER   = 0x10l
}; // enum buffer_type

enum buffer_usage {
    
}; // enum buffer_usage
 
class __scm_export(ogl) buffer_descriptor
{
    buffer_type     _type;
    buffer_usage    _usage;

    scm::size_t     _size;
}; // class buffer_descriptor

class __scm_export(ogl) buffer : public device_resource
{
public:
    virtual ~buffer();

protected:
    buffer(device& dev);

    const buffer_descriptor&    descriptor() const;

protected:
    buffer_descriptor           _descriptor;

    friend class scm::gl::device;
};

typedef scm::shared_ptr<buffer>  buffer_ptr;

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_BUFFER_H_INCLUDED_DEVICE_H_INCLUDED
