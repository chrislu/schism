
#include "transform_feedback.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>

namespace scm {
namespace gl {

stream_output_setup::stream_output_setup(const output_stream stream, const element& in_element)
  : _elements(static_cast<size_t>(OUTPUT_STREAM_COUNT))
  , _max_used_stream(0)
{
    assert(_elements.size() == OUTPUT_STREAM_COUNT);
    insert(stream, in_element);
}

stream_output_setup::stream_output_setup(const output_stream stream, const buffer_ptr& out_buffer, const size_t out_buffer_offset)
  : _elements(static_cast<size_t>(OUTPUT_STREAM_COUNT))
  , _max_used_stream(0)
{
    assert(_elements.size() == OUTPUT_STREAM_COUNT);
    insert(stream, out_buffer, out_buffer_offset);
}

stream_output_setup::~stream_output_setup()
{
}

stream_output_setup&
stream_output_setup::operator()(const output_stream stream, const element& in_element)
{
    insert(stream, in_element);
    return *this;
}

stream_output_setup&
stream_output_setup::operator()(const output_stream stream, const buffer_ptr& out_buffer, const size_t out_buffer_offset)
{
    insert(stream, out_buffer, out_buffer_offset);
    return *this;
}

void
stream_output_setup::insert(const output_stream stream, const element& in_element)
{
    assert(0 <= stream && stream < _elements.size());
    _elements[stream] = in_element;
    _max_used_stream  = math::max<unsigned>(_max_used_stream, stream);
}

void
stream_output_setup::insert(const output_stream stream, const buffer_ptr& out_buffer, const size_t out_buffer_offset)
{
    assert(0 <= stream && stream < _elements.size());
    _elements[stream] = element(out_buffer, out_buffer_offset);
    _max_used_stream  = math::max<unsigned>(_max_used_stream, stream);
}

unsigned
stream_output_setup::max_used_stream() const
{
    return _max_used_stream;
}

bool
stream_output_setup::empty() const
{
    return _elements.empty();
}

const stream_output_setup::element&
stream_output_setup::operator[](const output_stream stream) const
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
                                       const stream_output_setup&   in_stream_out_setup)
  : render_device_child(in_device),
    _stream_out_setup(in_stream_out_setup)
{
    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        const opengl::gl3_core& glapi = in_device.opengl3_api();
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
    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        const opengl::gl3_core& glapi = parent_device().opengl3_api();

        assert(0 != object_id());
        glapi.glDeleteTransformFeedbacks(1, &(context_bindable_object::_gl_object_id));

        gl_assert(glapi, leaving transform_feedback::~transform_feedback());
    }
}

const buffer_ptr&
transform_feedback::stream_out_buffer(const output_stream stream) const
{
    return _stream_out_setup[stream].first;
}

const buffer_ptr&
transform_feedback::operator[](const output_stream stream) const
{
    return stream_out_buffer(stream);
}

const stream_output_setup&
transform_feedback::stream_out_setup() const
{
    return _stream_out_setup;
}

void
transform_feedback::bind(render_context& in_context) const
{
    assert(state().ok());

    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        assert(object_id() != 0);

        const opengl::gl3_core& glapi = in_context.opengl_api();
        
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

    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        assert(object_id() != 0);

        const opengl::gl3_core& glapi = in_context.opengl_api();
        
        glapi.glBindTransformFeedback(object_target(), 0);

        gl_assert(glapi, transform_feedback::unbind() after glBindTransformFeedback());
    }
    else {
        unbind_stream_out_buffers(in_context);
    }

    gl_assert(in_context.opengl_api(), leaving transform_feedback:unbind());
}

bool
transform_feedback::initialize_transform_feedback_object(const render_device& in_device)
{
    if (_stream_out_setup.empty()) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        // GL4.x MAX_TRANSFORM_FEEDBACK_BUFFERS
        if (_stream_out_setup.max_used_stream() >= static_cast<unsigned>(in_device.capabilities()._max_transform_feedback_buffers)) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return false;
        }
    }
    else {
        // GL3.x MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS
        if (_stream_out_setup.max_used_stream() >= static_cast<unsigned>(in_device.capabilities()._max_transform_feedback_separate_attribs)) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return false;
        }
    }

    const render_context_ptr    context = in_device.main_context();

    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        const opengl::gl3_core& glapi = in_device.opengl3_api();
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
    const opengl::gl3_core& glapi = in_context.opengl_api();

    for (unsigned bind_index = 0; bind_index <= _stream_out_setup.max_used_stream(); ++bind_index) {
        const buffer_ptr& cur_buffer = _stream_out_setup[static_cast<output_stream>(bind_index)].first;
        const size_t      cur_offset = _stream_out_setup[static_cast<output_stream>(bind_index)].second;

        if (cur_buffer) {
            cur_buffer->bind_range(in_context, BIND_TRANSFORM_FEEDBACK_BUFFER, bind_index, cur_offset, 0);
        }
        assert(cur_buffer->ok());
    }
}

void
transform_feedback::unbind_stream_out_buffers(render_context& in_context) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    for (unsigned bind_index = 0; bind_index <= _stream_out_setup.max_used_stream(); ++bind_index) {
        const buffer_ptr& cur_buffer = _stream_out_setup[static_cast<output_stream>(bind_index)].first;

        if (cur_buffer) {
            cur_buffer->unbind_range(in_context, BIND_TRANSFORM_FEEDBACK_BUFFER, bind_index);
        }
        assert(cur_buffer->ok());
    }
}


} // namespace gl
} // namespace scm
