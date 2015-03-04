
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_BUFFER_H_INCLUDED
#define SCM_GL_CORE_BUFFER_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/context_bindable_object.h>
#include <scm/gl_core/render_device/device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class render_device;
class render_context;

struct __scm_export(gl_core) buffer_desc {
    buffer_desc() : _bindings(BIND_UNKNOWN), _usage(USAGE_STATIC_DRAW), _size(0) {}
    buffer_desc(buffer_binding b, buffer_usage u, scm::size_t s) : _bindings(b), _usage(u), _size(s) {}

    buffer_binding  _bindings;
    buffer_usage    _usage;
    scm::size_t     _size;
}; // struct buffer_desc

class __scm_export(gl_core) buffer : public context_bindable_object, public render_device_resource
{
public:
    virtual ~buffer();

    uint64                      native_handle() const { return _native_handle; }
    bool                        native_handle_resident() const { return _native_handle_resident; }

    const buffer_desc&          descriptor() const;
    void                        print(std::ostream& os) const;

protected:
    buffer(render_device&       ren_dev,
           const buffer_desc&   in_desc,
           const void*          initial_data);

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
                                    const access_mode   in_access);
    void*                       map_range(const render_context& in_context,
                                          scm::size_t           in_offset,
                                          scm::size_t           in_size,
                                          const access_mode     in_access);
    bool                        unmap(const render_context& in_context);


    bool                        buffer_data(const render_device& ren_dev,
                                            const buffer_desc&   in_desc,
                                            const void*          initial_data);
    bool                        buffer_sub_data(const render_device& ren_dev,
                                                scm::size_t          offset,
                                                scm::size_t          size,
                                                const void*          data);

    bool                        clear_buffer_data(const render_context& in_context,
                                                        data_format     in_format,
                                                  const void*           in_data);
    bool                        clear_buffer_sub_data(const render_context& in_context,
                                                            data_format     in_format,
                                                            scm::size_t     in_offset,
                                                            scm::size_t     in_size,
                                                      const void*           in_data);

    bool                        get_buffer_sub_data(const render_context& in_context,
                                                    scm::size_t           offset,
                                                    scm::size_t           size,
                                                    void*const            data);

    bool                        copy_buffer_data(const render_context& in_context,
                                                 const buffer&         in_src_buffer,
                                                       scm::size_t     in_dst_offset,
                                                       scm::size_t     in_src_offset,
                                                       scm::size_t     in_size);

    bool                        make_resident(const render_context&    in_context,
                                              const access_mode        in_access);

    bool                        make_non_resident(const render_context& in_context);

protected:
    buffer_desc                 _descriptor;

    bool                        _mapped;
    scm::size_t                 _mapped_interval_offset;
    scm::size_t                 _mapped_interval_length;

    uint64                      _native_handle;
    bool                        _native_handle_resident;

    friend class render_device;
    friend class render_context;
    friend class frame_buffer;
    friend class vertex_array;
    friend class texture_buffer;
    friend class transform_feedback;
};

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_BUFFER_H_INCLUDED
