
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "vertex_array.h"

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>
#include <scm/gl_core/shader_objects/program.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>

namespace scm {
namespace gl {

vertex_array::vertex_array(      render_device&           ren_dev,
                           const vertex_format&           in_vert_format,
                           const std::vector<buffer_ptr>& in_attrib_buffers,
                           const program_ptr&             in_program)
  : render_device_child(ren_dev),
    _vertex_format(in_vert_format)
{
    const opengl::gl_core& glapi = ren_dev.opengl_api();
    util::gl_error          glerror(glapi);

    glapi.glGenVertexArrays(1, &(context_bindable_object::_gl_object_id));
    if (0 == object_id()) {
        state().set(object_state::OS_BAD);
    }
    else {
        context_bindable_object::_gl_object_target  = 0; // no explicit binding point
        context_bindable_object::_gl_object_binding = GL_VERTEX_ARRAY_BINDING;

        if (build_buffer_slots(ren_dev, in_vert_format, in_attrib_buffers, in_program)) {
            initialize_array_object(ren_dev);
        }
    }
    gl_assert(glapi, leaving vertex_array::vertex_array());
}

vertex_array::~vertex_array()
{
    const opengl::gl_core& glapi = parent_device().opengl_api();

    assert(0 != object_id());
    glapi.glDeleteVertexArrays(1, &(context_bindable_object::_gl_object_id));

    gl_assert(glapi, leaving vertex_array::~vertex_array());
}

void
vertex_array::bind(render_context& ren_ctx) const
{
    assert(object_id() != 0);
    assert(state().ok());

    const opengl::gl_core& glapi = ren_ctx.opengl_api();

    glapi.glBindVertexArray(object_id());

    gl_assert(glapi, leaving vertex_array:bind());
}

void
vertex_array::unbind(render_context& ren_ctx) const
{
    assert(state().ok());

    const opengl::gl_core& glapi = ren_ctx.opengl_api();

    glapi.glBindVertexArray(0);

    gl_assert(glapi, leaving vertex_array:unbind());
}

bool
vertex_array::build_buffer_slots(const render_device&           in_ren_dev,
                                 const vertex_format&           in_vert_format,
                                 const std::vector<buffer_ptr>& in_attrib_buffers,
                                 const program_ptr&             in_program)
{
    assert(in_ren_dev.capabilities()._max_vertex_attributes > 0);

    scm::size_t num_input_streams     = in_attrib_buffers.size();
    int         max_vertex_attributes = in_ren_dev.capabilities()._max_vertex_attributes;
    if (num_input_streams > max_vertex_attributes) {
        return (false);
    }

    vertex_format::element_array::const_iterator e     = in_vert_format.elements().begin();
    vertex_format::element_array::const_iterator e_end = in_vert_format.elements().end();

    for (; e != e_end; ++e) {
        if (0 > e->_buffer_stream || e->_buffer_stream > num_input_streams) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return (false);
        }

        buffer_slot&        slot     = _buffer_slots[e->_buffer_stream];
        buffer_slot_element new_elmt = buffer_slot_element(*e, slot._size);

        if (!new_elmt._attrib_name.empty()) {
            if (in_program) {
                new_elmt._attrib_location   = in_program->attribute_location(e->_attrib_name);
                new_elmt._generic_attribute = false;
            }
            else {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                return (false);
            }
        }
        if (0 > new_elmt._attrib_location || new_elmt._attrib_location > max_vertex_attributes) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return (false);
        }
        if (   0 == (in_attrib_buffers[new_elmt._buffer_stream]->descriptor()._bindings & BIND_VERTEX_BUFFER)
            && 0 == (in_attrib_buffers[new_elmt._buffer_stream]->descriptor()._bindings & BIND_TRANSFORM_FEEDBACK_BUFFER)
            && 0 == (in_attrib_buffers[new_elmt._buffer_stream]->descriptor()._bindings & BIND_STORAGE_BUFFER)
            && 0 == (in_attrib_buffers[new_elmt._buffer_stream]->descriptor()._bindings & BIND_TEXTURE_BUFFER)) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return (false);
        }

        assert(in_attrib_buffers[new_elmt._buffer_stream]);

        slot._buffer = in_attrib_buffers[new_elmt._buffer_stream];
        slot._elements.push_back(new_elmt);
        slot._size += size_of_type(e->_type);
    }

    return (true);
}

bool
vertex_array::initialize_array_object(const render_device& ren_dev)
{
    assert(object_id() != 0);

    const opengl::gl_core& glapi = ren_dev.opengl_api();
    util::gl_error          glerror(glapi);

    buffer_slot_map::const_iterator slot     = _buffer_slots.begin();
    buffer_slot_map::const_iterator slot_end = _buffer_slots.end();

#if SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    //glapi.glBindVertexArray(0); // W T F ?!?!
    //glapi.glBindVertexArray(object_id());
#else
    util::vertex_array_binding_guard binding_guard(glapi);
    glapi.glBindVertexArray(object_id());
#endif // !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS

    for(; slot != slot_end; ++slot) {
        buffer_slot::slot_element_array::const_iterator elmt     = slot->second._elements.begin();
        buffer_slot::slot_element_array::const_iterator elmt_end = slot->second._elements.end();
        const buffer_ptr& buf = slot->second._buffer;

        if (!SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            //util::buffer_binding_guard bguard(glapi, GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING);
            // TODO check an warn if buffer target not compatible with array buffer
            glapi.glBindBuffer(GL_ARRAY_BUFFER, buf->object_id());
        }

        for (; elmt != elmt_end; ++elmt) {
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                glapi.glEnableVertexArrayAttribEXT(object_id(), elmt->_attrib_location);

                if (   is_integer_type(elmt->_type)
                    && elmt->_integer_handling == INT_PURE) {
                    glapi.glVertexArrayVertexAttribIOffsetEXT(object_id(), buf->object_id(),
                                                              elmt->_attrib_location,
                                                              components(elmt->_type),
                                                              util::gl_base_type(elmt->_type),
                                                              elmt->_stride,
                                                              elmt->_offset);
                }
                else {
    gl_assert(glapi, leaving vertex_array::initialize_array_object());
                    bool normalize = (is_integer_type(elmt->_type) && (elmt->_integer_handling == INT_FLOAT_NORMALIZE));
                    glapi.glVertexArrayVertexAttribOffsetEXT(object_id(), buf->object_id(),
                                                             elmt->_attrib_location,
                                                             components(elmt->_type),
                                                             util::gl_base_type(elmt->_type),
                                                             normalize,
                                                             elmt->_stride,
                                                             elmt->_offset);
    gl_assert(glapi, leaving vertex_array::initialize_array_object());
                }
                if (glerror) {
                    state().set(glerror.to_object_state());
                    return (false);
                }
            }
            else { // SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
                glapi.glEnableVertexAttribArray(elmt->_attrib_location);

                if (   is_integer_type(elmt->_type)
                    && elmt->_integer_handling == INT_PURE) {
                        glapi.glVertexAttribIPointer(elmt->_attrib_location,
                                                     components(elmt->_type),
                                                     util::gl_base_type(elmt->_type),
                                                     elmt->_stride,
                                                     BUFFER_OFFSET(elmt->_offset));
                }
                else {
                    bool normalize = (is_integer_type(elmt->_type) && (elmt->_integer_handling == INT_FLOAT_NORMALIZE));
                    glapi.glVertexAttribPointer(elmt->_attrib_location,
                                                components(elmt->_type),
                                                util::gl_base_type(elmt->_type),
                                                normalize,
                                                elmt->_stride,
                                                BUFFER_OFFSET(elmt->_offset));
                }
                if (glerror) {
                    state().set(glerror.to_object_state());
                    return (false);
                }
            }
        }

        if (!SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }

    gl_assert(glapi, leaving vertex_array::initialize_array_object());

    return (true);
}

} // namespace gl
} // namespace scm
