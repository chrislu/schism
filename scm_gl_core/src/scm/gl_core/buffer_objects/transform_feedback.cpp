
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "transform_feedback.h"

#include <cassert>
#include <limits>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>

namespace scm {
namespace gl {

stream_output_setup::stream_output_setup()
{
}

stream_output_setup::stream_output_setup(const element& in_element)
{
    insert(in_element);
}

stream_output_setup::stream_output_setup(const buffer_ptr& out_buffer, const size_t out_buffer_offset)
{
    insert(out_buffer, out_buffer_offset);
}

stream_output_setup::~stream_output_setup()
{
}

stream_output_setup&
stream_output_setup::operator()(const element& in_element)
{
    insert(in_element);
    return *this;
}

stream_output_setup&
stream_output_setup::operator()(const buffer_ptr& out_buffer, const size_t out_buffer_offset)
{
    insert(out_buffer, out_buffer_offset);
    return *this;
}

void
stream_output_setup::insert(const element& in_element)
{
    _elements.push_back(in_element);
}

void
stream_output_setup::insert(const buffer_ptr& out_buffer, const size_t out_buffer_offset)
{
    _elements.push_back(element(out_buffer, out_buffer_offset));
}

int
stream_output_setup::used_streams() const
{
    assert(_elements.size() < (std::numeric_limits<int>::max)());
    return static_cast<int>(_elements.size());
}

bool
stream_output_setup::empty() const
{
    return _elements.empty();
}

const stream_output_setup::element&
stream_output_setup::operator[](const int stream) const
{
    assert(0 <= stream && stream < _elements.size());
    return _elements[stream];
}

bool
stream_output_setup::operator==(const stream_output_setup& rhs) const
{
    return _elements == rhs._elements;
}

bool
stream_output_setup::operator!=(const stream_output_setup& rhs) const
{
    return _elements != rhs._elements;
}

transform_feedback::transform_feedback(      render_device&         in_device,
                                       const stream_output_setup&   in_setup)
  : render_device_child(in_device)
  , _stream_out_setup(in_setup)
  , _active(false)
  , _captured_topology(PRIMITIVE_POINTS)
{
    if (SCM_GL_CORE_OPENGL_TYPE >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        const opengl::gl_core& glapi = in_device.opengl_api();
        util::gl_error          glerror(glapi);

        glapi.glGenTransformFeedbacks(1, &(context_bindable_object::_gl_object_id));
        if (0 == object_id()) {
            state().set(object_state::OS_BAD);
        }
        else {
            context_bindable_object::_gl_object_target  = GL_TRANSFORM_FEEDBACK;
            context_bindable_object::_gl_object_binding = GL_TRANSFORM_FEEDBACK_BINDING;

            if (!initialize_transform_feedback_object(in_device)) {
                // the state is set in the functions to more detailed error states
            }
        }
        gl_assert(glapi, leaving transform_feedback::transform_feedback());
    }
}

transform_feedback::~transform_feedback()
{
    if (SCM_GL_CORE_OPENGL_TYPE >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        const opengl::gl_core& glapi = parent_device().opengl_api();

        assert(0 != object_id());
        glapi.glDeleteTransformFeedbacks(1, &(context_bindable_object::_gl_object_id));

        gl_assert(glapi, leaving transform_feedback::~transform_feedback());
    }
}

const buffer_ptr&
transform_feedback::stream_out_buffer(const int stream) const
{
    assert(0 <= stream && stream < _stream_out_setup.used_streams());
    return _stream_out_setup[stream].first;
}

const buffer_ptr&
transform_feedback::operator[](const int stream) const
{
    return stream_out_buffer(stream);
}

const stream_output_setup&
transform_feedback::stream_out_setup() const
{
    return _stream_out_setup;
}

bool
transform_feedback::active() const
{
    return _active;
}

primitive_type
transform_feedback::captured_topology() const
{
    return _captured_topology;
}

void
transform_feedback::bind(render_context& in_context) const
{
    assert(state().ok());

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        assert(object_id() != 0);

        const opengl::gl_core& glapi = in_context.opengl_api();
        
        glapi.glBindTransformFeedback(object_target(), object_id());
        
        gl_assert(glapi, transform_feedback::bind() after glBindTransformFeedback());
    }
    else {
        bind_stream_out_buffers(in_context);
    }

    gl_assert(in_context.opengl_api(), leaving transform_feedback:bind());
}

void
transform_feedback::unbind(render_context& in_context) const
{
    assert(state().ok());

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        assert(object_id() != 0);

        const opengl::gl_core& glapi = in_context.opengl_api();

        glapi.glBindTransformFeedback(object_target(), 0);

        gl_assert(glapi, transform_feedback::unbind() after glBindTransformFeedback());
    }
    else {
        unbind_stream_out_buffers(in_context);
    }

    gl_assert(in_context.opengl_api(), leaving transform_feedback:unbind());
}

void
transform_feedback::begin(render_context& in_context, primitive_type in_topology_mode) 
{
    assert(state().ok());
    gl_assert(in_context.opengl_api(), entering transform_feedback:begin());

    const opengl::gl_core& glapi = in_context.opengl_api();

    bind(in_context);
    if (!active()) {
        glapi.glBeginTransformFeedback(util::gl_primitive_type(in_topology_mode));
    }
    gl_assert(in_context.opengl_api(), leaving transform_feedback:begin());

    _active            = true;
    _captured_topology = in_topology_mode;

    gl_assert(in_context.opengl_api(), leaving transform_feedback:begin());
}

void
transform_feedback::end(render_context& in_context)
{
    assert(state().ok());
    gl_assert(in_context.opengl_api(), entering transform_feedback:end());

    const opengl::gl_core& glapi = in_context.opengl_api();

    if (active()) {
        glapi.glEndTransformFeedback();
    }
    unbind(in_context);

    _active            = false;

    gl_assert(in_context.opengl_api(), leaving transform_feedback:end());
}

bool
transform_feedback::initialize_transform_feedback_object(const render_device& in_device)
{
    if (_stream_out_setup.empty()) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        // GL4.x MAX_TRANSFORM_FEEDBACK_BUFFERS
        if (_stream_out_setup.used_streams() > in_device.capabilities()._max_transform_feedback_buffers) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return false;
        }
    }
    else {
        // GL3.x MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS
        if (_stream_out_setup.used_streams() > in_device.capabilities()._max_transform_feedback_separate_attribs) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return false;
        }
    }

    const render_context_ptr    context = in_device.main_context();

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        const opengl::gl_core& glapi = in_device.opengl_api();
        util::gl_error          glerror(glapi);

        util::transform_feedback_binding_guard guard(glapi, object_target(), object_binding());
        glapi.glBindTransformFeedback(object_target(), object_id());

        bind_stream_out_buffers(*context);
        if (glerror) {
            state().set(glerror.to_object_state());
            unbind(*context);

            return false;
        }
    }

    return true;
}

void
transform_feedback::bind_stream_out_buffers(render_context& in_context) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    for (int bind_index = 0; bind_index < _stream_out_setup.used_streams(); ++bind_index) {
        const buffer_ptr& cur_buffer = _stream_out_setup[bind_index].first;
        const size_t      cur_offset = _stream_out_setup[bind_index].second;

        if (cur_buffer) {
            cur_buffer->bind_range(in_context, BIND_TRANSFORM_FEEDBACK_BUFFER, bind_index, cur_offset, 0);
        }
        assert(cur_buffer->ok());
    }
}

void
transform_feedback::unbind_stream_out_buffers(render_context& in_context) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    for (int bind_index = 0; bind_index < _stream_out_setup.used_streams(); ++bind_index) {
        const buffer_ptr& cur_buffer = _stream_out_setup[bind_index].first;

        if (cur_buffer) {
            cur_buffer->unbind_range(in_context, BIND_TRANSFORM_FEEDBACK_BUFFER, bind_index);
        }
        assert(cur_buffer->ok());
    }
}

} // namespace gl
} // namespace scm
