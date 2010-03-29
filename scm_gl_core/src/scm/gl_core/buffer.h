
#ifndef SCM_GL_CORE_BUFFER_H_INCLUDED
#define SCM_GL_CORE_BUFFER_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_core/render_device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class render_device;
class render_context;

class __scm_export(gl_core) buffer : public render_device_resource
{
public:
    enum binding_type {
        TYPE_UNKNOWN             = 0x00,
        TYPE_VERTEX_BUFFER       = 0x01,
        TYPE_INDEX_BUFFER        = 0x02,
        TYPE_PIXEL_BUFFER        = 0x04,
        TYPE_UNIFORM_BUFFER      = 0x08,
        TYPE_STREAM_OUT_BUFFER   = 0x10
    }; // enum buffer_type

    enum usage_type {
        // write once
        STATIC_DRAW,        // GPU r,  CPU
        STATIC_READ,        // GPU     CPU r
        STATIC_COPY,        // GPU rw, CPU
        // low write frequency
        STREAM_DRAW,        // GPU r,  CPU w
        STREAM_READ,        // GPU w,  CPU r
        STREAM_COPY,        // GPU rw, CPU
        // high write frequency
        DYNAMIC_DRAW,       // GPU r,  CPU w
        DYNAMIC_READ,       // GPU w,  CPU r
        DYNAMIC_COPY        // GPU rw, CPU
    }; // enum buffer_usage

    struct descriptor_type {
        descriptor_type() : _bindings(TYPE_UNKNOWN), _usage(STATIC_DRAW), _size(0) {}
        descriptor_type(binding_type b, usage_type u, scm::size_t s) : _bindings(b), _usage(u), _size(s) {}

        binding_type    _bindings;
        usage_type      _usage;
        scm::size_t     _size;
    }; // struct descriptor

public:
    virtual ~buffer();

    const descriptor_type&      descriptor() const;
    void                        print(std::ostream& os) const;

protected:
    buffer(render_device&           ren_dev,
           const descriptor_type&   buffer_desc,
           const void*              initial_data);

    bool                        buffer_data(      render_device&     ren_dev,
                                            const descriptor_type&   buffer_desc,
                                            const void*              initial_data);
    bool                        buffer_sub_data(render_device&  ren_dev,
                                                scm::size_t     offset,
                                                scm::size_t     size,
                                                const void*     data);

protected:
    descriptor_type             _descriptor;
    unsigned                    _gl_buffer_id;

    scm::size_t                 _mapped_interval_offset;
    scm::size_t                 _mapped_interval_length;

    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_BUFFER_H_INCLUDED
