
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_VERTEX_ARRAY_H_INCLUDED
#define SCM_GL_CORE_VERTEX_ARRAY_H_INCLUDED

#include <map>
#include <vector>

#include <scm/core/numeric_types.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>
#include <scm/gl_core/render_device/context_bindable_object.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) vertex_array : public context_bindable_object, public render_device_child
{
protected:
    struct buffer_slot_element : public vertex_format::element {
        buffer_slot_element(const vertex_format::element& e, int o) : vertex_format::element(e), _generic_attribute(true), _offset(o) {}

        bool    _generic_attribute;
        int     _offset;
    }; //  struct buffer_slot_element
    struct buffer_slot {
        typedef std::vector<buffer_slot_element> slot_element_array;
        buffer_slot() : _size(0) {}

        buffer_ptr          _buffer;
        slot_element_array  _elements;
        int                 _size;
    }; // struct buffer_slot
    typedef std::map<int, buffer_slot>  buffer_slot_map;

public:
    virtual ~vertex_array();

protected:
    vertex_array(      render_device&           ren_dev,
                 const vertex_format&           in_vert_format,
                 const std::vector<buffer_ptr>& in_attrib_buffers,
                 const program_ptr&             in_program = program_ptr());

    void                bind(render_context& ren_ctx) const;
    void                unbind(render_context& ren_ctx) const;
    bool                build_buffer_slots(const render_device&           in_ren_dev,
                                           const vertex_format&           in_vert_format,
                                           const std::vector<buffer_ptr>& in_attrib_buffers,
                                           const program_ptr&             in_program);
    bool                initialize_array_object(const render_device& ren_dev);

protected:
    vertex_format       _vertex_format;
    buffer_slot_map     _buffer_slots;

    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
}; // class vertex_array

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_VERTEX_ARRAY_H_INCLUDED
