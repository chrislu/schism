
#ifndef SCM_GL_CORE_BUFFER_H_INCLUDED
#define SCM_GL_CORE_BUFFER_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class render_device;
class render_context;

class __scm_export(gl_core) buffer : public render_device_resource
{
public:
    struct descriptor_type {
        descriptor_type() : _bindings(BIND_UNKNOWN), _usage(USAGE_STATIC_DRAW), _size(0) {}
        descriptor_type(buffer_binding b, buffer_usage u, scm::size_t s) : _bindings(b), _usage(u), _size(s) {}

        buffer_binding  _bindings;
        buffer_usage    _usage;
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

    void                        bind(render_context& ren_ctx, buffer_binding target) const;
    void                        unbind(render_context& ren_ctx, buffer_binding target) const;

    void                        bind_range(render_context&   in_context,
                                           buffer_binding    in_target,
                                           const unsigned    in_index,
                                           const scm::size_t in_offset,
                                           const scm::size_t in_size);
    void                        unbind_range(render_context&   in_context,
                                             buffer_binding    in_target,
                                             const unsigned    in_index);

    void*                       map(const render_context& in_context,
                                    const buffer_access   in_access);
    void*                       map_range(const render_context& in_context,
                                          scm::size_t           in_offset,
                                          scm::size_t           in_size,
                                          const buffer_access   in_access);
    bool                        unmap(const render_context& in_context);


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

    friend class render_device;
    friend class render_context;
    friend class vertex_array;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_BUFFER_H_INCLUDED
