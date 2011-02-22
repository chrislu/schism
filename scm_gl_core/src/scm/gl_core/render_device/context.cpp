
#include "context.h"

#include <scm/gl_core/config.h>
#include <scm/gl_core/object_state.h>
#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/frame_buffer_objects.h>
#include <scm/gl_core/query_objects.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>

#include <scm/gl_core/log.h>

namespace scm {
namespace gl {
namespace detail {
} // namespace detail

render_context::index_buffer_binding::index_buffer_binding()
  : _primitive_topology(PRIMITIVE_POINT_LIST), _index_data_type(TYPE_UINT), _index_data_offset(0)
{
}

bool
render_context::index_buffer_binding::operator==(const index_buffer_binding& rhs) const
{
    return    (_index_buffer       == rhs._index_buffer)
           && (_primitive_topology == rhs._primitive_topology)
           && (_index_data_type    == rhs._index_data_type)
           && (_index_data_offset  == rhs._index_data_offset);
}

bool
render_context::index_buffer_binding::operator!=(const index_buffer_binding& rhs) const
{
    return    (_index_buffer       != rhs._index_buffer)
           || (_primitive_topology != rhs._primitive_topology)
           || (_index_data_type    != rhs._index_data_type)
           || (_index_data_offset  != rhs._index_data_offset);
}

bool
render_context::buffer_binding::operator==(const buffer_binding& rhs) const
{
    return    (_buffer == rhs._buffer)
           && (_offset == rhs._offset)
           && (_size   == rhs._size);
}

bool
render_context::buffer_binding::operator!=(const buffer_binding& rhs) const
{
    return    (_buffer != rhs._buffer)
           || (_offset != rhs._offset)
           || (_size   != rhs._size);
}

render_context::image_unit_binding::image_unit_binding()
  : _texture_image()
  , _format(FORMAT_NULL)
  , _access(ACCESS_READ_WRITE)
  , _level(0)
  , _layer(0)
{
}

bool
render_context::image_unit_binding::operator==(const image_unit_binding& rhs) const
{
    return    (_texture_image == rhs._texture_image)
           && (_format        == rhs._format)
           && (_access        == rhs._access)
           && (_level         == rhs._level)
           && (_layer         == rhs._layer);
}

bool
render_context::image_unit_binding::operator!=(const image_unit_binding& rhs) const
{
    return    (_texture_image != rhs._texture_image)
           || (_format        != rhs._format)
           || (_access        != rhs._access)
           || (_level         != rhs._level)
           || (_layer         != rhs._layer);
}

render_context::binding_state_type::binding_state_type()
  : _stencil_ref_value(0)
  , _line_width(1.0f)
  , _point_size(1.0f)
  , _blend_color(math::vec4f(1.0f, 1.0f, 1.0f, 1.0f))
  , _default_framebuffer_target(FRAMEBUFFER_BACK)
  , _viewports(viewport(math::vec2ui(0, 0), math::vec2ui(10, 10)))
{
}

render_context::render_context(render_device& in_device)
  : render_device_child(in_device)
  , _opengl_api_core(in_device.opengl3_api())
{
    const opengl::gl3_core& glapi = opengl_api();

    _default_depth_stencil_state = in_device.create_depth_stencil_state(depth_stencil_state_desc());
    _default_depth_stencil_state->force_apply(*this, _current_state._stencil_ref_value);
    _current_state._depth_stencil_state = _default_depth_stencil_state;
    _applied_state._depth_stencil_state = _default_depth_stencil_state;

    _default_rasterizer_state = in_device.create_rasterizer_state(rasterizer_state_desc());
    _default_rasterizer_state->force_apply(*this, _current_state._line_width, _current_state._point_size);
    _current_state._rasterizer_state = _default_rasterizer_state;
    _applied_state._rasterizer_state = _default_rasterizer_state;

    _default_blend_state = in_device.create_blend_state(blend_state_desc());
    _default_blend_state->force_apply(*this, _current_state._blend_color);
    _current_state._blend_state = _default_blend_state;
    _applied_state._blend_state = _default_blend_state;

    _current_state._texture_units.resize(in_device.capabilities()._max_texture_image_units);
    _applied_state._texture_units.resize(in_device.capabilities()._max_texture_image_units);
    if (   glapi.extension_EXT_shader_image_load_store
        && in_device.capabilities()._max_image_units > 0) {
        _current_state._image_units.resize(in_device.capabilities()._max_image_units);
        _applied_state._image_units.resize(in_device.capabilities()._max_image_units);
    }

    _current_state._active_uniform_buffers.resize(in_device.capabilities()._max_uniform_buffer_bindings);
    _applied_state._active_uniform_buffers.resize(in_device.capabilities()._max_uniform_buffer_bindings);

    glapi.glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glapi.glPixelStorei(GL_PACK_ALIGNMENT, 1);

    _debug_synchronous_reporting = true;

    _active_transform_feedback_topology_mode = PRIMITIVE_POINTS;

    gl_assert(glapi, leaving render_context::render_context());
}

render_context::~render_context()
{
    _default_depth_stencil_state.reset();
}

const opengl::gl3_core&
render_context::opengl_api() const
{
    return (_opengl_api_core);
}

void
render_context::apply()
{
    gl_assert(opengl_api(), entering render_context::apply());

    assert(state().ok());

    apply_texture_units();
    apply_image_units();
    apply_frame_buffer();
    apply_vertex_input();
    apply_state_objects();
    apply_uniform_buffer_bindings();
    apply_program();

    assert(state().ok());
}

void
render_context::reset()
{
    reset_image_units();
    reset_texture_units();
    reset_framebuffer();
    reset_vertex_input();
    reset_state_objects();
    reset_uniform_buffers();
    reset_program();
}

void
render_context::flush()
{
    //apply();
    //apply_texture_units();
    //apply_image_units();
    //apply_frame_buffer();
    //apply_vertex_input();

    // bind the vertex_array
    //if (_current_state._vertex_array != _applied_state._vertex_array) {
    //    if (_current_state._vertex_array) {
    //        glout() << "_current_state._vertex_array->bind(*this);" << log::end;
    //        _current_state._vertex_array->bind(*this);
    //    }
    //    else {
    //        _applied_state._vertex_array->unbind(*this);
    //        glout() << "_applied_state._vertex_array->unbind(*this);" << log::end;
    //    }
    //    _applied_state._vertex_array = _current_state._vertex_array;
    //}

    //opengl_api().glBindVertexArray(0);

    //if (_current_state._index_buffer_binding != _applied_state._index_buffer_binding) {
    //    if (_current_state._index_buffer_binding._index_buffer != _applied_state._index_buffer_binding._index_buffer) {
    //        if (_current_state._index_buffer_binding._index_buffer) {
    //            glout() << "_current_state._index_buffer_binding._index_buffer->bind(*this, BIND_INDEX_BUFFER);" << log::end;
    //            _current_state._index_buffer_binding._index_buffer->bind(*this, BIND_INDEX_BUFFER);
    //        }
    //        else {
    //            glout() << "_applied_state._index_buffer_binding._index_buffer->unbind(*this, BIND_INDEX_BUFFER);" << log::end;
    //            _applied_state._index_buffer_binding._index_buffer->unbind(*this, BIND_INDEX_BUFFER);
    //        }
    //    }
    //    _applied_state._index_buffer_binding = _current_state._index_buffer_binding;
    //}


    //apply_state_objects();
    //apply_uniform_buffer_bindings();
    //apply_program();

    apply();
    opengl_api().glFlush();
    //opengl_api().glFinish();
}

void
render_context::sync()
{
    flush();
    opengl_api().glFinish();
    opengl_api().glMemoryBarrierEXT(GL_ALL_BARRIER_BITS_EXT);
}

// debug api //////////////////////////////////////////////////////////////////////////////////
void
render_context::register_debug_callback(const debug_output_ptr& f)
{
    const opengl::gl3_core& glapi = opengl_api();

    //if (!glapi.is_supported("GL_ARB_debug_output")) {
    if (!glapi.extension_ARB_debug_output) {
        glout() << log::warning << "render_context::register_debug_callback(): "
                << "no debug context present (GL_ARB_debug_output unsupported), ignoring debug output." << log::end;
        return;
    }

    assert(f);

    if (_debug_outputs.empty()) {
        // register the debug callback
        glapi.glDebugMessageCallbackARB(&render_context::gl_debug_callback, this);
        gl_assert(glapi, render_context::register_debug_callback() after glDebugMessageCallbackARB());

        if (_debug_synchronous_reporting) {
            glapi.glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        }
        else {
            glapi.glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        }
    }

    _debug_outputs.insert(f);

    //glapi.glDebugMessageInsertARB(GL_DEBUG_SOURCE_APPLICATION_ARB, GL_DEBUG_TYPE_ERROR_ARB, 0, GL_DEBUG_SEVERITY_HIGH_ARB, 4, "test");
    //glapi.glEnable(GL_VERTEX_SHADER);

    gl_assert(glapi, leaving render_context::register_debug_callback());

    //glout() << "registered callback" << log::end;
}

void
render_context::unregister_debug_callback(const debug_output_ptr& f)
{
    const opengl::gl3_core& glapi = opengl_api();

    //if (!glapi.is_supported("GL_ARB_debug_output")) {
    if (!glapi.extension_ARB_debug_output) {
        glout() << log::warning << "render_context::unregister_debug_callback(): "
                << "no debug context present (GL_ARB_debug_output unsupported), ignoring debug output." << log::end;
        return;
    }

    assert(f);

    boost::unordered_set<debug_output_ptr>::const_iterator dbg_out = _debug_outputs.find(f);

    if (dbg_out != _debug_outputs.end()) {
        _debug_outputs.quick_erase(dbg_out);
    }

    if (_debug_outputs.empty()) {
        // unregister the gl callback
        glapi.glDebugMessageCallbackARB(0, 0);
        glapi.glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
        gl_assert(glapi, render_context::unregister_debug_callback() after glDebugMessageCallbackARB());
    }
}

const std::string
render_context::retrieve_debug_log() const
{
    const opengl::gl3_core& glapi = opengl_api();

    // if (!glapi.is_supported("GL_ARB_debug_output")) {
    if (!glapi.extension_ARB_debug_output) {
        glout() << log::warning << "render_context::retrieve_debug_log(): "
                << "no debug context present (GL_ARB_debug_output unsupported), ignoring debug output." << log::end;
        return (std::string(""));
    }

    //glapi.glDebugMessageInsertARB(GL_DEBUG_SOURCE_APPLICATION_ARB, GL_DEBUG_TYPE_ERROR_ARB, 0, GL_DEBUG_SEVERITY_HIGH_ARB, 5, "test");
    //glapi.glEnable(GL_VERTEX_SHADER);

    int num_messages     = 0;
    int max_mmessage_len = 0;

    glapi.glGetIntegerv(GL_DEBUG_LOGGED_MESSAGES_ARB, &num_messages);
    glapi.glGetIntegerv(GL_MAX_DEBUG_MESSAGE_LENGTH_ARB, &max_mmessage_len);

    scoped_array<unsigned>  sources(new unsigned[num_messages]);
    scoped_array<unsigned>  types(new unsigned[num_messages]);
    scoped_array<unsigned>  ids(new unsigned[num_messages]);
    scoped_array<unsigned>  severities(new unsigned[num_messages]);
    scoped_array<int>       lengths(new int[num_messages]);
    scoped_array<char>      messages(new char[num_messages * max_mmessage_len]);

    unsigned ret_messages = glapi.glGetDebugMessageLogARB(num_messages, num_messages * max_mmessage_len,
                                                          sources.get(), types.get(), ids.get(), severities.get(), lengths.get(), messages.get());

    std::stringstream output;
    int message_pos = 0;
    for (unsigned i = 0; i < ret_messages; ++i) {
        output << "<source: " << debug_source_string(util::gl_to_debug_source(sources[i]))
               << ", id: " << ids[i]
               << ", type: " << debug_type_string(util::gl_to_debug_type(types[i]))
               << ", severity: " << debug_severity_string(util::gl_to_debug_severity(severities[i])) << "> "
               << &messages[message_pos] << std::endl;
        message_pos += lengths[i];
    }

    return (output.str());
}

void
render_context::synchronous_reporting(bool e)
{
    if (e != _debug_synchronous_reporting) {
        _debug_synchronous_reporting = e;

        const opengl::gl3_core& glapi = opengl_api();
        if (glapi.extension_ARB_debug_output) {
            if (_debug_synchronous_reporting) {
                glapi.glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
            }
            else {
                glapi.glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
            }
        }
    }
}

bool
render_context::synchronous_reporting() const
{
    return (_debug_synchronous_reporting);
}

/*static*/
void
render_context::gl_debug_callback(unsigned src, unsigned type, unsigned id, unsigned severity,
                                  int msg_length, const char* msg, void* user_param)
{
    if (0 != user_param) {
        static_cast<render_context*>(user_param)->gl_debug_dispatch(src, type, severity, msg_length, msg);
    }
}

void
render_context::gl_debug_dispatch(unsigned src, unsigned type, unsigned severity, int msg_length, const char* msg)
{
    boost::unordered_set<debug_output_ptr>::iterator b = _debug_outputs.begin();
    boost::unordered_set<debug_output_ptr>::iterator e = _debug_outputs.end();

    std::string dbg_message;

    if (b != e) { // not empty
        dbg_message.assign(msg, msg_length);
    }

    for (; b != e; ++b) {
        (*b)->operator()(util::gl_to_debug_source(src), util::gl_to_debug_type(type), util::gl_to_debug_severity(severity), dbg_message);
    }
}

// buffer api /////////////////////////////////////////////////////////////////////////////////
void*
render_context::map_buffer(const buffer_ptr&  in_buffer,
                           const access_mode in_access) const
{
    void* return_value = in_buffer->map(*this, in_access);

    if (   (0 == return_value)
        || (!in_buffer->ok())) {
        SCM_GL_DGB("render_context::map_buffer(): error mapping buffer ('" << in_buffer->state().state_string() << "')");
    }

    return (return_value);
}

void*
render_context::map_buffer_range(const buffer_ptr&   in_buffer,
                                 scm::size_t         in_offset,
                                 scm::size_t         in_size,
                                 const access_mode in_access) const
{
    void* return_value = in_buffer->map_range(*this, in_offset, in_size, in_access);

    if (   (0 == return_value)
        || (!in_buffer->ok())) {
        SCM_GL_DGB("render_context::map_buffer_range(): error mapping buffer range ('" << in_buffer->state().state_string() << "')");
    }

    return (return_value);
}

bool
render_context::unmap_buffer(const buffer_ptr& in_buffer) const
{
    bool return_value = in_buffer->unmap(*this);

    if (   (false == return_value)
        || (!in_buffer->ok())) {
        SCM_GL_DGB("render_context::unmap_buffer(): error unmapping buffer ('" << in_buffer->state().state_string() << "')");
    }

    return (return_value);
}

bool
render_context::orphane_buffer(const buffer_ptr& in_buffer) const
{
    bool return_value = in_buffer->buffer_data(parent_device(), in_buffer->descriptor(), 0);

    if (   (false == return_value)
        || (!in_buffer->ok())) {
        SCM_GL_DGB("render_context::orphane_buffer(): error orphaning buffer ('" << in_buffer->state().state_string() << "')");
    }

    return return_value;
}

void
render_context::bind_uniform_buffer(const buffer_ptr& in_buffer,
                                    const unsigned    in_bind_point,
                                    const scm::size_t in_offset,
                                    const scm::size_t in_size)
{
     assert(in_bind_point < _current_state._active_uniform_buffers.size());
    _current_state._active_uniform_buffers[in_bind_point]._buffer = in_buffer;
    _current_state._active_uniform_buffers[in_bind_point]._offset = in_offset;
    _current_state._active_uniform_buffers[in_bind_point]._size   = in_size;
}

void
render_context::set_uniform_buffers(const buffer_binding_array& in_buffers)
{
    _current_state._active_uniform_buffers = in_buffers;
}

const render_context::buffer_binding_array&
render_context::current_uniform_buffers() const
{
    return (_current_state._active_uniform_buffers);
}

void
render_context::reset_uniform_buffers()
{
    std::fill(_current_state._active_uniform_buffers.begin(),
              _current_state._active_uniform_buffers.end(),
              buffer_binding());
}

void
render_context::bind_unpack_buffer(const buffer_ptr& in_buffer)
{
    if (_unpack_buffer != in_buffer) {
        if (in_buffer) {
            in_buffer->bind(*this, BIND_PIXEL_UNPACK_BUFFER);
        }
        else {
            _unpack_buffer->unbind(*this, BIND_PIXEL_UNPACK_BUFFER);
        }
        _unpack_buffer = in_buffer;
    }
}

const buffer_ptr&
render_context::current_unpack_buffer() const
{
    return (_unpack_buffer);
}

void
render_context::bind_vertex_array(const vertex_array_ptr& in_vertex_array)
{
    _current_state._vertex_array = in_vertex_array;
}

const vertex_array_ptr&
render_context::current_vertex_array() const
{
    return (_current_state._vertex_array);
}

void
render_context::bind_index_buffer(const buffer_ptr& in_buffer, const primitive_topology in_topology, const data_type in_index_type, const scm::size_t in_offset)
{
    _current_state._index_buffer_binding._index_buffer        = in_buffer;
    _current_state._index_buffer_binding._primitive_topology  = in_topology;
    _current_state._index_buffer_binding._index_data_type     = in_index_type;
    _current_state._index_buffer_binding._index_data_offset   = in_offset;
}

void
render_context::current_index_buffer(buffer_ptr& out_buffer, primitive_topology& out_topology, data_type& out_index_type, scm::size_t& out_offset) const
{
    out_buffer      = _current_state._index_buffer_binding._index_buffer;
    out_topology    = _current_state._index_buffer_binding._primitive_topology;
    out_index_type  = _current_state._index_buffer_binding._index_data_type;
    out_offset      = _current_state._index_buffer_binding._index_data_offset;
}

void
render_context::set_index_buffer_binding(const index_buffer_binding& in_index_buffer_binding)
{
    _current_state._index_buffer_binding = in_index_buffer_binding;
}

const render_context::index_buffer_binding&
render_context::current_index_buffer_binding() const
{
    return (_current_state._index_buffer_binding);
}

void
render_context::reset_vertex_input()
{
    _current_state._vertex_array         = vertex_array_ptr();
    _current_state._index_buffer_binding = index_buffer_binding();
}

void
render_context::start_transform_feedback()
{
    if (_active_transform_feedback) {
        if (!_active_transform_feedback->active()) {
            _active_transform_feedback->begin(*this, _active_transform_feedback_topology_mode);
        }
    }
}

void
render_context::begin_transform_feedback(const transform_feedback_ptr& in_transform_feedback, primitive_type in_topology_mode)
{
    if (_active_transform_feedback) {
        glerr() << log::warning
                << "render_context::begin_transform_feedback(): active transform feedback object present, "
                << "overriding active transform feedback." << log::end;
        assert(_active_transform_feedback->active());
        _active_transform_feedback->end(*this); // unbind implicit
    }
    //in_transform_feedback->begin(*this, in_topology_mode); // bind implicit
    _active_transform_feedback               = in_transform_feedback;
    _active_transform_feedback_topology_mode = in_topology_mode;

    gl_assert(opengl_api(), leaving render_context::begin_transform_feedback());
}

void
render_context::end_transform_feedback()
{
    if (!_active_transform_feedback) {
        glerr() << log::warning
                << "render_context::end_transform_feedback(): no transform feedback currently active." << log::end;
    }
    else {
        if (_active_transform_feedback->active()) {
            _active_transform_feedback->end(*this);
        }
        _active_transform_feedback = transform_feedback_ptr();
    }

    gl_assert(opengl_api(), leaving render_context::end_transform_feedback());
}

const transform_feedback_ptr&
render_context::active_transform_feedback() const
{
    return _active_transform_feedback;
}

void
render_context::draw_transform_feedback(const primitive_topology in_topology, const transform_feedback_ptr& in_transform_feedback, int stream)
{
    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        const opengl::gl3_core& glapi = opengl_api();

        // check if feedback object is currently active and give warning
        if (active_transform_feedback() == in_transform_feedback) {
            SCM_GL_DGB(   "render_context::draw_transform_feedback(): "
                       << "transform feedback object is currently active, results from previous operation used.");
        }

        pre_draw_setup();

        if (stream < 0) {
            glapi.glDrawTransformFeedback(util::gl_primitive_topology(in_topology), in_transform_feedback->object_id());
        }
        else {
            if (stream > parent_device().capabilities()._max_vertex_streams) {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                SCM_GL_DGB(    "render_context::draw_transform_feedback(): error invalid vertex stream (>"
                           <<  parent_device().capabilities()._max_vertex_streams
                           << ") passed"
                           << "('" << state().state_string() << "')");
                return;
            }
            glapi.glDrawTransformFeedbackStream(util::gl_primitive_topology(in_topology), in_transform_feedback->object_id(), stream);
        }

        post_draw_setup();

    }
    else {
        glerr() << log::error
                << "render_context::draw_transform_feedback(): "
                << "the draw_transform_feedback functionality is only available using scm_gl_core with OpenGL4.x capabilities enabled on a OpenGL4.x context."
                << log::end;
    }
}

void
render_context::draw_arrays(const primitive_topology in_topology, const int in_first_index, const int in_count)
{
    const opengl::gl3_core& glapi = opengl_api();

    if (   (0 > in_first_index)
        || (0 > in_count)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        SCM_GL_DGB("render_context::draw_arrays(): error invalid count or start index (< 0) " << "('" << state().state_string() << "')");
        return;
    }

    pre_draw_setup();

    glapi.glDrawArrays(util::gl_primitive_topology(in_topology), in_first_index, in_count);

    gl_assert(glapi, leaving render_context::draw_arrays());
}

void
render_context::draw_elements(const int in_count, const int in_start_index, const int in_base_vertex)
{
    const opengl::gl3_core& glapi = opengl_api();

    if (!util::is_vaild_index_type(_applied_state._index_buffer_binding._index_data_type)) {
        state().set(object_state::OS_ERROR_INVALID_ENUM);
        return;
    }
    if (   (0 > in_count)
        || (0 > in_start_index)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        SCM_GL_DGB("render_context::draw_elements(): error invalid count or start index (< 0) " << "('" << state().state_string() << "')");
        return;
    }

    pre_draw_setup();

    glapi.glDrawElementsBaseVertex(
        util::gl_primitive_topology(_applied_state._index_buffer_binding._primitive_topology),
        in_count,
        util::gl_base_type(_applied_state._index_buffer_binding._index_data_type),
        (char*)0 + _applied_state._index_buffer_binding._index_data_offset + size_of_type(_applied_state._index_buffer_binding._index_data_type) * in_start_index,
        in_base_vertex);

    post_draw_setup();

    gl_assert(glapi, leaving render_context::draw_elements());
}

void
render_context::pre_draw_setup()
{
    const opengl::gl3_core& glapi = opengl_api();

    gl_assert(glapi, entering render_context::pre_draw());
    assert(state().ok());

    start_transform_feedback();

    if (_applied_state._program->rasterization_discard()) {
        glapi.glEnable(GL_RASTERIZER_DISCARD);
    }

    if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400) {
        int patch_control_points = primitive_patch_control_points(_applied_state._index_buffer_binding._primitive_topology);
        if (0 != patch_control_points) {
            glapi.glPatchParameteri(GL_PATCH_VERTICES, patch_control_points);

            gl_assert(glapi, render_context::pre_draw_setup() after glPatchParameteri);
        }
    }

    gl_assert(glapi, leaving render_context::pre_draw());
}

void
render_context::post_draw_setup()
{
    const opengl::gl3_core& glapi = opengl_api();

    gl_assert(glapi, entering render_context::post_draw());
    
    if (_applied_state._program->rasterization_discard()) {
        glapi.glDisable(GL_RASTERIZER_DISCARD);
    }

    gl_assert(glapi, leaving render_context::post_draw());
}

void
render_context::apply_vertex_input()
{
    //if (!_current_state._vertex_array) {
    //    state().set(object_state::OS_ERROR_INVALID_VALUE);
    //    return;
    //}

    // bind the vertex_array
    if (_current_state._vertex_array != _applied_state._vertex_array) {
        if (_current_state._vertex_array) {
            _current_state._vertex_array->bind(*this);
        }
        else {
            _applied_state._vertex_array->unbind(*this);
        }
        _applied_state._vertex_array = _current_state._vertex_array;
    }

    if (_current_state._index_buffer_binding != _applied_state._index_buffer_binding) {
        if (_current_state._index_buffer_binding._index_buffer != _applied_state._index_buffer_binding._index_buffer) {
            if (_current_state._index_buffer_binding._index_buffer) {
                _current_state._index_buffer_binding._index_buffer->bind(*this, BIND_INDEX_BUFFER);
            }
            else {
                _applied_state._index_buffer_binding._index_buffer->unbind(*this, BIND_INDEX_BUFFER);
            }
        }
        _applied_state._index_buffer_binding = _current_state._index_buffer_binding;
    }
    gl_assert(opengl_api(), leaving render_context::apply_vertex_input());
}

void
render_context::apply_uniform_buffer_bindings()
{
    for (int i = 0; i < _current_state._active_uniform_buffers.size(); ++i) {
        const buffer_binding&   cubb = _current_state._active_uniform_buffers[i];
        buffer_binding&         aubb = _applied_state._active_uniform_buffers[i];

        if (cubb != aubb) {
            if (cubb._buffer) {
                cubb._buffer->bind_range(*this, BIND_UNIFORM_BUFFER, i, cubb._offset, cubb._size);
                assert(cubb._buffer->ok());
            }
            else {
                aubb._buffer->unbind_range(*this, BIND_UNIFORM_BUFFER, i);
            }
            aubb = cubb;
        }
    }
}

// shader api /////////////////////////////////////////////////////////////////////////////////
void
render_context::bind_program(const program_ptr& in_program)
{
    _current_state._program = in_program;
}

const program_ptr&
render_context::current_program() const
{
    return (_current_state._program);
}

void
render_context::reset_program()
{
    _current_state._program = program_ptr();
}

void
render_context::apply_program()
{
    if (!_current_state._program) {
        // TODO some kind of read state validation before draw, the can be a state with no program bound
        //state().set(object_state::OS_ERROR_INVALID_VALUE);
        return;
    }
    if (!_current_state._program->ok()) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return;
    }

    // bind the program
    if (_current_state._program != _applied_state._program) {
        _current_state._program->bind(*this);
        _applied_state._program = _current_state._program;
    }

    // bind uniforms
    _applied_state._program->bind_uniforms(*this);

    gl_assert(opengl_api(), leaving render_context::apply_program());
}

// texture api ////////////////////////////////////////////////////////////////////////////////
void
render_context::bind_texture(const texture_ptr&       in_texture_image,
                             const sampler_state_ptr& in_sampler_state,
                             const unsigned           in_unit)
{
    assert(in_unit < _current_state._texture_units.size());

    if (in_unit >= _current_state._texture_units.size()) {
        return;
    }

    _current_state._texture_units[in_unit]._texture_image = in_texture_image;
    _current_state._texture_units[in_unit]._sampler_state = in_sampler_state;
}

void
render_context::set_texture_unit_state(const texture_unit_array& in_texture_units)
{
    _current_state._texture_units = in_texture_units;
}

const render_context::texture_unit_array&
render_context::current_texture_unit_state() const
{
    return (_current_state._texture_units);
}

void
render_context::reset_texture_units()
{
    std::fill(_current_state._texture_units.begin(),
              _current_state._texture_units.end(),
              texture_unit_binding());
}

void
render_context::bind_image(const texture_ptr&       in_texture_image,
                                 data_format        in_format,
                                 access_mode        in_access,
                                 unsigned           in_unit,
                                 int                in_level,
                                 int                in_layer)
{
    assert(in_unit < _current_state._image_units.size());

    if (in_unit < _current_state._image_units.size()) {
        image_unit_binding& cur_binding = _current_state._image_units[in_unit];

        cur_binding._texture_image = in_texture_image;
        cur_binding._format        = in_format;
        cur_binding._access        = in_access;
        cur_binding._level         = in_level;
        cur_binding._layer         = in_layer;
    }
    else {
        if (SCM_GL_DEBUG) {
            glerr() << log::error
                    << "render_context::bind_image(): "
                    << "error, no image units available (EXT_shader_image_load_store unavailable)."
                    << log::end;
        }
    }
}

void
render_context::set_image_unit_state(const image_unit_array& in_imageunits)
{
    _current_state._image_units = in_imageunits;
}

const render_context::image_unit_array&
render_context::current_image_unit_state() const
{
    return _current_state._image_units;
}

void
render_context::reset_image_units()
{
    std::fill(_current_state._image_units.begin(),
              _current_state._image_units.end(),
              image_unit_binding());
}

bool
render_context::update_sub_texture(const texture_image_ptr& in_texture,
                                   const texture_region&    in_region,
                                   const unsigned           in_level,
                                   const data_format        in_data_format,
                                   const size_t             in_offset)
{
    assert(_unpack_buffer);
    if (!in_texture->image_sub_data(*this, in_region, in_level, in_data_format, BUFFER_OFFSET(in_offset))) {
        glerr() << log::error
                << "render_context::update_sub_texture(): "
                << "error during sub texture update (check update region)."
                << log::end;
        return (false);
    }
    return (true);
}

bool
render_context::update_sub_texture(const texture_image_ptr& in_texture,
                                   const texture_region&    in_region,
                                   const unsigned           in_level,
                                   const data_format        in_data_format,
                                   const void*const         in_data)
{
    assert(!_unpack_buffer);
    if (!in_texture->image_sub_data(*this, in_region, in_level, in_data_format, in_data)) {
        glerr() << log::error
                << "render_context::update_sub_texture(): "
                << "error during sub texture update (check update region)."
                << log::end;
        return (false);
    }
    return (true);
}

void
render_context::apply_texture_units()
{
    for (int u = 0; u < _current_state._texture_units.size(); ++u) {
        texture_ptr&        cti = _current_state._texture_units[u]._texture_image;
        texture_ptr&        ati = _applied_state._texture_units[u]._texture_image;

        if (cti != ati) {
            if (cti) {
                cti->bind(*this, u);
            }
            else {
                ati->unbind(*this, u);
            }
            ati = cti;
        }

        sampler_state_ptr&  css = _current_state._texture_units[u]._sampler_state;
        sampler_state_ptr&  ass = _applied_state._texture_units[u]._sampler_state;
        if (css != ass) {
            if (css) {
                css->bind(*this, u);
            }
            else {
                ass->unbind(*this, u);
            }
            ass = css;
        }
    }
    gl_assert(opengl_api(), leaving render_context::apply_texture_units());
}

void
render_context::apply_image_units()
{
    for (int u = 0; u < _current_state._image_units.size(); ++u) {
        const image_unit_binding& cub = _current_state._image_units[u];
        image_unit_binding&       aub = _applied_state._image_units[u];

        if (cub != aub) {
            if (cub._texture_image) {
                cub._texture_image->bind_image(*this, u, cub._format, cub._access, cub._level, cub._layer);
            }
            else {
                aub._texture_image->unbind_image(*this, u);
            }
        }
        aub = cub;
    }

    gl_assert(opengl_api(), leaving render_context::apply_image_units());
}

// frame buffer api ///////////////////////////////////////////////////////////////////////////
void
render_context::set_frame_buffer(const frame_buffer_ptr& in_frame_buffer)
{
    _current_state._draw_framebuffer = in_frame_buffer;
}

void
render_context::set_default_frame_buffer(const frame_buffer_target in_target)
{
    _current_state._draw_framebuffer           = frame_buffer_ptr();
    _current_state._default_framebuffer_target = in_target;
}

const frame_buffer_ptr&
render_context::current_frame_buffer() const
{
    return (_current_state._draw_framebuffer);
}

const frame_buffer_target
render_context::current_default_frame_buffer_target() const
{
    return (_current_state._default_framebuffer_target);
}

void
render_context::set_viewport(const viewport& in_vp)
{
    _current_state._viewports = viewport_array(in_vp);
}

void
render_context::set_viewports(const viewport_array& in_vp)
{
    if (in_vp.size() > parent_device().capabilities()._max_viewports) {
        glerr() << log::warning
                << "render_context::set_viewports(): exceeded max number of supported viewports "
                << "(max_viewports: " << parent_device().capabilities()._max_viewports << ")" << log::end;
    }
    _current_state._viewports = in_vp;
}

const viewport_array&
render_context::current_viewports() const
{
    return (_current_state._viewports);
}

void
render_context::reset_framebuffer()
{
    set_default_frame_buffer(FRAMEBUFFER_BACK);
}

void
render_context::clear_color_buffer(const frame_buffer_ptr& in_frame_buffer,
                                   const unsigned          in_buffer,
                                   const math::vec4f&      in_clear_color) const
{
    const opengl::gl3_core& glapi = opengl_api();

    const blend_ops_array& abops = _applied_state._blend_state->descriptor()._blend_ops;
    if (1 == abops.size()) {
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(true, true, true, true);
        }

        in_frame_buffer->clear_color_buffer(*this, in_buffer, in_clear_color);
        
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(util::masked(abops[0]._write_mask, COLOR_RED),  util::masked(abops[0]._write_mask, COLOR_GREEN),
                              util::masked(abops[0]._write_mask, COLOR_BLUE), util::masked(abops[0]._write_mask, COLOR_ALPHA));
        }
    }
    else if (in_buffer < abops.size()) {
        if (abops[in_buffer]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(in_buffer, true, true, true, true);
        }
        
        in_frame_buffer->clear_color_buffer(*this, in_buffer, in_clear_color);
        
        if (abops[in_buffer]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(in_buffer, util::masked(abops[in_buffer]._write_mask, COLOR_RED),  util::masked(abops[in_buffer]._write_mask, COLOR_GREEN),
                                          util::masked(abops[in_buffer]._write_mask, COLOR_BLUE), util::masked(abops[in_buffer]._write_mask, COLOR_ALPHA));
        }
    }
    else {
        glerr() << log::warning
                << "render_context::clear_color_buffer(): no blend state defined for color buffer " << in_buffer << log::end;
    }

    gl_assert(glapi, leaving render_context::clear_color_buffer());
}

void
render_context::clear_color_buffer(const frame_buffer_ptr& in_frame_buffer,
                                   const unsigned          in_buffer,
                                   const math::vec4i&      in_clear_color) const
{
    const opengl::gl3_core& glapi = opengl_api();

    const blend_ops_array& abops = _applied_state._blend_state->descriptor()._blend_ops;
    if (1 == abops.size()) {
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(true, true, true, true);
        }

        in_frame_buffer->clear_color_buffer(*this, in_buffer, in_clear_color);
        
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(util::masked(abops[0]._write_mask, COLOR_RED),  util::masked(abops[0]._write_mask, COLOR_GREEN),
                              util::masked(abops[0]._write_mask, COLOR_BLUE), util::masked(abops[0]._write_mask, COLOR_ALPHA));
        }
    }
    else if (in_buffer < abops.size()) {
        if (abops[in_buffer]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(in_buffer, true, true, true, true);
        }
        
        in_frame_buffer->clear_color_buffer(*this, in_buffer, in_clear_color);
        
        if (abops[in_buffer]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(in_buffer, util::masked(abops[in_buffer]._write_mask, COLOR_RED),  util::masked(abops[in_buffer]._write_mask, COLOR_GREEN),
                                          util::masked(abops[in_buffer]._write_mask, COLOR_BLUE), util::masked(abops[in_buffer]._write_mask, COLOR_ALPHA));
        }
    }
    else {
        glerr() << log::warning
                << "render_context::clear_color_buffer(): no blend state defined for color buffer " << in_buffer << log::end;
    }

    gl_assert(glapi, leaving render_context::clear_color_buffer());
}

void
render_context::clear_color_buffer(const frame_buffer_ptr& in_frame_buffer,
                                   const unsigned          in_buffer,
                                   const math::vec4ui&     in_clear_color)
{
    const opengl::gl3_core& glapi = opengl_api();

    const blend_ops_array& abops = _applied_state._blend_state->descriptor()._blend_ops;
    if (1 == abops.size()) {
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(true, true, true, true);
        }

        in_frame_buffer->clear_color_buffer(*this, in_buffer, in_clear_color);
        
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(util::masked(abops[0]._write_mask, COLOR_RED),  util::masked(abops[0]._write_mask, COLOR_GREEN),
                              util::masked(abops[0]._write_mask, COLOR_BLUE), util::masked(abops[0]._write_mask, COLOR_ALPHA));
        }
    //glout() << "ok" << log::end;
    ////apply_texture_units();
    ////apply_image_units();
    ////apply_frame_buffer();
    ////apply_vertex_input();
    ////apply_state_objects();
    ////apply_uniform_buffer_bindings();
    ////apply_program();
    //apply();
    //glapi.glFlush();
    }
    else if (in_buffer < abops.size()) {
        if (abops[in_buffer]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(in_buffer, true, true, true, true);
        }
        
        in_frame_buffer->clear_color_buffer(*this, in_buffer, in_clear_color);
        
        if (abops[in_buffer]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(in_buffer, util::masked(abops[in_buffer]._write_mask, COLOR_RED),  util::masked(abops[in_buffer]._write_mask, COLOR_GREEN),
                                          util::masked(abops[in_buffer]._write_mask, COLOR_BLUE), util::masked(abops[in_buffer]._write_mask, COLOR_ALPHA));
        }
    }
    else {
        glerr() << log::warning
                << "render_context::clear_color_buffer(): no blend state defined for color buffer " << in_buffer << log::end;
    }

    //apply();
    ////glapi.glBindVertexArray(0);
    //glapi.glFlush();

    gl_assert(glapi, leaving render_context::clear_color_buffer());
}

void
render_context::clear_color_buffers(const frame_buffer_ptr& in_frame_buffer,
                                    const math::vec4f&      in_clear_color) const
{
    const opengl::gl3_core& glapi = opengl_api();

    const blend_ops_array& abops = _applied_state._blend_state->descriptor()._blend_ops;
    if (1 == abops.size()) {
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(true, true, true, true);
        }

        in_frame_buffer->clear_color_buffers(*this, in_clear_color);

        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(util::masked(abops[0]._write_mask, COLOR_RED),  util::masked(abops[0]._write_mask, COLOR_GREEN),
                              util::masked(abops[0]._write_mask, COLOR_BLUE), util::masked(abops[0]._write_mask, COLOR_ALPHA));
        }
    }
    else {
        for (int i = 0; i < abops.size(); ++i) {
            if (abops[i]._write_mask != COLOR_ALL) {
                glapi.glColorMaski(i, true, true, true, true);
            }
        }
        in_frame_buffer->clear_color_buffers(*this, in_clear_color);
        for (int i = 0; i < abops.size(); ++i) {
            if (abops[i]._write_mask != COLOR_ALL) {
                glapi.glColorMaski(i, util::masked(abops[i]._write_mask, COLOR_RED),  util::masked(abops[i]._write_mask, COLOR_GREEN),
                                      util::masked(abops[i]._write_mask, COLOR_BLUE), util::masked(abops[i]._write_mask, COLOR_ALPHA));
            }
        }
    }

    gl_assert(glapi, leaving render_context::clear_color_buffers());
}

void
render_context::clear_depth_stencil_buffer(const frame_buffer_ptr& in_frame_buffer,
                                           const float             in_clear_depth,
                                           const int               in_clear_stencil) const
{
    const opengl::gl3_core& glapi = opengl_api();

    if (   (false == _applied_state._depth_stencil_state->_descriptor._depth_mask)
        || (0     == _applied_state._depth_stencil_state->_descriptor._stencil_wmask)) {
        glapi.glDepthMask(true);
        glapi.glStencilMask(true);

        in_frame_buffer->clear_depth_stencil_buffer(*this, in_clear_depth, in_clear_stencil);
        
        glapi.glDepthMask(_applied_state._depth_stencil_state->_descriptor._depth_mask);
        glapi.glStencilMask(_applied_state._depth_stencil_state->_descriptor._stencil_wmask);
    }
    else {
        in_frame_buffer->clear_depth_stencil_buffer(*this, in_clear_depth, in_clear_stencil);
    }

    gl_assert(glapi, leaving render_context::clear_depth_stencil_buffer());
}

void
render_context::clear_default_color_buffer(const frame_buffer_target in_target,
                                           const math::vec4f&        in_clear_color) const
{
    const opengl::gl3_core& glapi = opengl_api();

    if (_applied_state._draw_framebuffer) {
        _applied_state._draw_framebuffer->unbind(*this);
    }
    if (_applied_state._default_framebuffer_target != in_target) {
        glapi.glDrawBuffer(util::gl_frame_buffer_target(in_target));
    }

    const blend_ops_array& abops = _applied_state._blend_state->descriptor()._blend_ops;
    if (1 == abops.size()) {
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(true, true, true, true);
        }
        
        if (true/*SCM_GL_CORE_USE_WORKAROUND_AMD*/) {
            glapi.glClearColor(in_clear_color.r, in_clear_color.g, in_clear_color.b, in_clear_color.a);
            glapi.glClear(GL_COLOR_BUFFER_BIT);
        }
        else {
            glapi.glClearBufferfv(GL_COLOR, 0, in_clear_color.data_array);
        }

        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMask(util::masked(abops[0]._write_mask, COLOR_RED),  util::masked(abops[0]._write_mask, COLOR_GREEN),
                              util::masked(abops[0]._write_mask, COLOR_BLUE), util::masked(abops[0]._write_mask, COLOR_ALPHA));
        }
    }
    else {
        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(0, true, true, true, true);
        }

        if (true/*SCM_GL_CORE_USE_WORKAROUND_AMD*/) {
            glapi.glClearColor(in_clear_color.r, in_clear_color.g, in_clear_color.b, in_clear_color.a);
            glapi.glClear(GL_COLOR_BUFFER_BIT);
        }
        else {
            glapi.glClearBufferfv(GL_COLOR, 0, in_clear_color.data_array);
        }

        if (abops[0]._write_mask != COLOR_ALL) {
            glapi.glColorMaski(0, util::masked(abops[0]._write_mask, COLOR_RED),  util::masked(abops[0]._write_mask, COLOR_GREEN),
                                  util::masked(abops[0]._write_mask, COLOR_BLUE), util::masked(abops[0]._write_mask, COLOR_ALPHA));
        }
    }

    if (_applied_state._default_framebuffer_target != in_target) {
        glapi.glDrawBuffer(util::gl_frame_buffer_target(_applied_state._default_framebuffer_target));
    }
    if (_applied_state._draw_framebuffer) {
        _applied_state._draw_framebuffer->bind(*this, FRAMEBUFFER_DRAW);
    }

    gl_assert(glapi, leaving render_context::clear_default_color_buffer());
}

void
render_context::clear_default_depth_stencil_buffer(const float in_clear_depth,
                                                   const int   in_clear_stencil) const
{
    const opengl::gl3_core& glapi = opengl_api();

    if (_applied_state._draw_framebuffer) {
        _applied_state._draw_framebuffer->unbind(*this);
    }

    if (   (false == _applied_state._depth_stencil_state->_descriptor._depth_mask)
        || (0     == _applied_state._depth_stencil_state->_descriptor._stencil_wmask)) {
        glapi.glDepthMask(true);
        glapi.glStencilMask(true);

        if (true/*SCM_GL_CORE_USE_WORKAROUND_AMD*/) {
            glapi.glClearDepth(in_clear_depth);
            glapi.glClearStencil(in_clear_stencil);
            glapi.glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        }
        else {
            //if (is_stencil_format(_current_depth_stencil_attachment._target->format())) {
                glapi.glClearBufferfi(GL_DEPTH_STENCIL, 0, in_clear_depth, in_clear_stencil);
            //}
            //else {
            //    glapi.glClearBufferfv(GL_DEPTH, 0, &in_clear_depth);
            //}
        }
        
        glapi.glDepthMask(_applied_state._depth_stencil_state->_descriptor._depth_mask);
        glapi.glStencilMask(_applied_state._depth_stencil_state->_descriptor._stencil_wmask);
    }
    else {
        if (true/*SCM_GL_CORE_USE_WORKAROUND_AMD*/) {
            glapi.glClearDepth(in_clear_depth);
            glapi.glClearStencil(in_clear_stencil);
            glapi.glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        }
        else {
            //if (is_stencil_format(_current_depth_stencil_attachment._target->format())) {
                glapi.glClearBufferfi(GL_DEPTH_STENCIL, 0, in_clear_depth, in_clear_stencil);
            //}
            //else {
            //    glapi.glClearBufferfv(GL_DEPTH, 0, &in_clear_depth);
            //}
        }
    }

    if (_applied_state._draw_framebuffer) {
        _applied_state._draw_framebuffer->bind(*this, FRAMEBUFFER_DRAW);
    }

    gl_assert(glapi, leaving render_context::clear_default_depth_stencil_buffer());
}

void
render_context::resolve_multi_sample_buffer(const frame_buffer_ptr& in_read_buffer,
                                            const frame_buffer_ptr& in_draw_buffer) const
{
    const opengl::gl3_core& glapi = opengl_api();

    if (_applied_state._draw_framebuffer != in_draw_buffer) {
        in_draw_buffer->apply_attachments(*this);
        in_draw_buffer->bind(*this, FRAMEBUFFER_DRAW);
    }
    if (_applied_state._read_framebuffer != in_read_buffer) {
        in_read_buffer->apply_attachments(*this);
        in_read_buffer->bind(*this, FRAMEBUFFER_READ);
    }

    math::vec2ui min_drawable_region = math::min(in_draw_buffer->drawable_region(),
                                                 in_read_buffer->drawable_region());

    gl_assert(glapi, render_context::resolve_multi_sample_buffer() before glBlitFramebuffer);

    glapi.glBlitFramebuffer(0, 0, min_drawable_region.x, min_drawable_region.y,
                            0, 0, min_drawable_region.x, min_drawable_region.y,
                            GL_COLOR_BUFFER_BIT,// | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
                            GL_NEAREST);

    gl_assert(glapi, render_context::resolve_multi_sample_buffer() after glBlitFramebuffer);

    if (_applied_state._read_framebuffer != in_read_buffer) {
        if (_applied_state._read_framebuffer) {
            _applied_state._read_framebuffer->bind(*this, FRAMEBUFFER_READ);
        }
        else {
            in_read_buffer->unbind(*this);
        }
    }
    if (_applied_state._draw_framebuffer != in_draw_buffer) {
        if (_applied_state._draw_framebuffer) {
            _applied_state._draw_framebuffer->bind(*this, FRAMEBUFFER_DRAW);
        }
        else {
            in_draw_buffer->unbind(*this);
        }
    }
    gl_assert(glapi, leaving render_context::resolve_multi_sample_buffer());
}

void
render_context::generate_mipmaps(const texture_image_ptr& in_texture) const
{
    const opengl::gl3_core& glapi = opengl_api();

    in_texture->generate_mipmaps(*this);

    gl_assert(glapi, leaving render_context::generate_mipmaps());
}

void
render_context::capture_color_buffer(const frame_buffer_ptr& in_frame_buffer,
                                     const unsigned          in_buffer,
                                     const texture_region&   in_region,
                                     const data_format       in_data_format,
                                     const buffer_ptr&       in_target_buffer,
                                     const size_t            in_offset)
{
    const opengl::gl3_core& glapi = opengl_api();

    in_frame_buffer->capture_color_buffer(*this, in_buffer, in_region, in_data_format, in_target_buffer, in_offset);

    gl_assert(glapi, leaving render_context::capture_color_buffer());
}

void
render_context::apply_frame_buffer()
{
    const opengl::gl3_core& glapi = opengl_api();

    if (_current_state._draw_framebuffer) {
        _current_state._draw_framebuffer->apply_attachments(*this);
        if (SCM_GL_DEBUG) {
            if (!_current_state._draw_framebuffer->check_completeness(*this)) {
                glerr() << log::error
                        << "render_context::apply_frame_buffer(): incomplete framebuffer "
                        << "('" << state().state_string() << "')." << log::end;
                return;
            }
        }
    }

    if (_current_state._draw_framebuffer != _applied_state._draw_framebuffer) {
        if (_current_state._draw_framebuffer) {
            _current_state._draw_framebuffer->bind(*this, FRAMEBUFFER_DRAW);
        }
        else {
            _applied_state._draw_framebuffer->unbind(*this);
        }
        _applied_state._draw_framebuffer = _current_state._draw_framebuffer;
    }
    if (!_applied_state._draw_framebuffer) { // we are on the default frame buffer
        if (_current_state._default_framebuffer_target != _applied_state._default_framebuffer_target) {
            glapi.glDrawBuffer(util::gl_frame_buffer_target(_current_state._default_framebuffer_target));
            _applied_state._default_framebuffer_target = _current_state._default_framebuffer_target;
        }
    }

    if (_current_state._viewports != _applied_state._viewports) {
        if (SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_410) {
            using namespace scm::math;
            int vp_array_size = parent_device().capabilities()._max_viewports;
            scoped_array<vec4f> vp_array(new vec4f[vp_array_size]);
            scoped_array<vec2d> dr_array(new vec2d[vp_array_size]);

            if (_current_state._viewports.size() == 1) {
                for (int i = 0; i < vp_array_size; ++i) {
                    vp_array[i] = vec4f(_current_state._viewports.viewports()[0]._position,
                                        _current_state._viewports.viewports()[0]._dimensions.x,
                                        _current_state._viewports.viewports()[0]._dimensions.y);
                    dr_array[i] = _current_state._viewports.viewports()[0]._depth_range;
                }
            }
            else {
                for (int i = 0; i < _current_state._viewports.size(); ++i) {
                    vp_array[i] = vec4f(_current_state._viewports.viewports()[i]._position,
                                        _current_state._viewports.viewports()[i]._dimensions.x,
                                        _current_state._viewports.viewports()[i]._dimensions.y);
                    dr_array[i] = _current_state._viewports.viewports()[i]._depth_range;
                }
                for (int i = static_cast<int>(_current_state._viewports.size()); i < vp_array_size; ++i) {
                    vp_array[i] = vec4f(0.0f, 0.0f, 100.0f, 100.0f);
                    dr_array[i] = vec2d(0.0, 1.0);
                }
            }
            glapi.glViewportArrayv(0, vp_array_size, vp_array[0].data_array);
            glapi.glDepthRangeArrayv(0, vp_array_size, dr_array[0].data_array);
        }
        else {
            if (SCM_GL_DEBUG) {
                if (_current_state._viewports.size() > 1) {
                    glerr() << log::warning
                            << "render_context::apply_frame_buffer(): viewport array not supported. "
                            << "applying first viewport "
                            << "(build gl_core with OpenGL 4.1 configuration for enabled support) "
                            << log::end;
                }
            }
            glapi.glViewport(static_cast<unsigned>(_current_state._viewports.viewports()[0]._position.x),
                             static_cast<unsigned>(_current_state._viewports.viewports()[0]._position.y),
                             static_cast<unsigned>(_current_state._viewports.viewports()[0]._dimensions.x),
                             static_cast<unsigned>(_current_state._viewports.viewports()[0]._dimensions.y));
            glapi.glDepthRange(_current_state._viewports.viewports()[0]._depth_range.x,
                               _current_state._viewports.viewports()[0]._depth_range.y);
        }
        _applied_state._viewports = _current_state._viewports;
    }

    gl_assert(glapi, leaving render_context::apply_frame_buffer());
}

// state api //////////////////////////////////////////////////////////////////////////////////
void
render_context::set_depth_stencil_state(const depth_stencil_state_ptr& in_ds_state, unsigned in_stencil_ref)
{
    _current_state._depth_stencil_state = in_ds_state;
    _current_state._stencil_ref_value   = in_stencil_ref;
}

const depth_stencil_state_ptr&
render_context::current_depth_stencil_state() const
{
    return (_current_state._depth_stencil_state);
}

unsigned
render_context::current_stencil_ref_value() const
{
    return (_current_state._stencil_ref_value);
}

void
render_context::set_rasterizer_state(const rasterizer_state_ptr& in_rs_state, float in_line_width, float in_point_size)
{
    _current_state._rasterizer_state = in_rs_state;
    _current_state._line_width       = in_line_width;
    _current_state._point_size       = in_point_size;
}

const rasterizer_state_ptr&
render_context::current_rasterizer_state() const
{
    return (_current_state._rasterizer_state);
}

float
render_context::current_line_width() const
{
    return (_current_state._line_width);
}

float
render_context::current_point_size() const
{
    return (_current_state._point_size);
}

void
render_context::set_blend_state(const blend_state_ptr& in_bl_state, const math::vec4f& in_blend_color)
{
    _current_state._blend_state = in_bl_state;
    _current_state._blend_color = in_blend_color;
}

const blend_state_ptr&
render_context::current_blend_state() const
{
    return (_current_state._blend_state);
}

const math::vec4f&
render_context::current_blend_color() const
{
    return (_current_state._blend_color);
}

void
render_context::reset_state_objects()
{
    _current_state._depth_stencil_state = _default_depth_stencil_state;
    _current_state._rasterizer_state    = _default_rasterizer_state;
    _current_state._blend_state         = _default_blend_state;
}

void
render_context::apply_state_objects()
{
    if (   (_current_state._depth_stencil_state != _applied_state._depth_stencil_state)
        || (_current_state._stencil_ref_value != _applied_state._stencil_ref_value)) {
        _current_state._depth_stencil_state->apply(*this, _current_state._stencil_ref_value,
                                                   *(_applied_state._depth_stencil_state), _applied_state._stencil_ref_value);
        _applied_state._depth_stencil_state = _current_state._depth_stencil_state;
        _applied_state._stencil_ref_value   = _current_state._stencil_ref_value;
    }

    if (   (_current_state._rasterizer_state != _applied_state._rasterizer_state)
        || (_current_state._line_width       != _applied_state._line_width)) {
        _current_state._rasterizer_state->apply(*this, _current_state._line_width, _current_state._point_size,
                                                *(_applied_state._rasterizer_state), _applied_state._line_width, _applied_state._point_size);
        _applied_state._rasterizer_state = _current_state._rasterizer_state;
        _applied_state._line_width       = _current_state._line_width;
        _applied_state._point_size       = _current_state._point_size;
    }

    if (   (_current_state._blend_state != _applied_state._blend_state)
        || (_current_state._blend_color != _applied_state._blend_color)) {
        _current_state._blend_state->apply(*this, _current_state._blend_color,
                                           *(_applied_state._blend_state), _applied_state._blend_color);
        _applied_state._blend_state = _current_state._blend_state;
        _applied_state._blend_color = _current_state._blend_color;
    }
    gl_assert(opengl_api(), leaving render_context::apply_state_objects());
}

// active queries /////////////////////////////////////////////////////////////////////////
void
render_context::begin_query(const query_ptr& in_query)
{
    indexed_query_id cur_query = std::make_pair(in_query->query_type(), in_query->index());

    if (_active_queries[cur_query]) {
        glerr() << log::warning
                << "render_context::begin_query(): query of type and index ("
                << std::hex << in_query->query_type() << ", " << std::dec << in_query->index() << ") "
                << "allready active, overriding query." << log::end;
    }
    in_query->begin(*this);
    _active_queries[cur_query] = in_query;

    gl_assert(opengl_api(), leaving render_context::begin_query());
}

void
render_context::end_query(const query_ptr& in_query)
{
    indexed_query_id cur_query = std::make_pair(in_query->query_type(), in_query->index());

    if (_active_queries[cur_query] != in_query) {
        glerr() << log::warning
                << "render_context::end_query(): this query of type and index ("
                << std::hex << in_query->query_type() << ", " << std::dec << in_query->index() << ") "
                << "not currently active active, result may be undefined." << log::end;
    }
    in_query->end(*this);
    _active_queries[cur_query] = query_ptr();

    gl_assert(opengl_api(), leaving render_context::end_query());
}

void
render_context::collect_query_results(const query_ptr& in_query)
{
    indexed_query_id cur_query = std::make_pair(in_query->query_type(), in_query->index());

    if (_active_queries[cur_query] == in_query) {
        glerr() << log::warning
                << "render_context::collect_query_results(): this query of type and index ("
                << std::hex << in_query->query_type() << ", " << std::dec << in_query->index() << ") "
                << "is currently active, the collected results may be undefined." << log::end;
    }
    in_query->collect(*this);
    gl_assert(opengl_api(), leaving render_context::collect_query_results());
}

} // namespace gl
} // namespace scm
