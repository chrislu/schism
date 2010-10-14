
#include "buffer.h"

#include <cassert>
#include <iostream>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {
namespace {
} // namespace 

buffer::buffer(render_device& ren_dev,
               const descriptor_type&   buffer_desc,
               const void*              initial_data)
  : render_device_resource(ren_dev),
    _descriptor(),
    _gl_buffer_id(0),
    _mapped_interval_offset(0),
    _mapped_interval_length(0)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();

    glapi.glGenBuffers(1, &_gl_buffer_id);
    if (0 == _gl_buffer_id) {
        state().set(object_state::OS_BAD);
    }
    else {
        buffer_data(ren_dev, buffer_desc, initial_data);
    }
    gl_assert(glapi, leaving buffer::buffer());
}

buffer::~buffer()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_buffer_id);
    glapi.glDeleteBuffers(1, &_gl_buffer_id);
    
    gl_assert(glapi, leaving buffer::~buffer());
}

void
buffer::bind(render_context& ren_ctx, buffer_binding target) const
{
    const opengl::gl3_core& glapi = ren_ctx.opengl_api();

    gl_assert(glapi, entering buffer::bind());

    assert(_gl_buffer_id != 0);
    assert(state().ok());

    glapi.glBindBuffer(util::gl_buffer_targets(target), _gl_buffer_id);

    gl_assert(glapi, leaving buffer::bind());
}

void
buffer::unbind(render_context& ren_ctx, buffer_binding target) const
{
    const opengl::gl3_core& glapi = ren_ctx.opengl_api();

    gl_assert(glapi, entering buffer::unbind());

    glapi.glBindBuffer(util::gl_buffer_targets(target), 0);

    gl_assert(glapi, leaving buffer:unbind());
}

void
buffer::bind_range(render_context&   in_context,
                   buffer_binding    in_target,
                   const unsigned    in_index,
                   const scm::size_t in_offset,
                   const scm::size_t in_size)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering buffer::bind_range());

    assert(_gl_buffer_id != 0);
    assert(state().ok());

    if (   (0 > in_offset)
        || (0 > in_size)
        || (_descriptor._size < (in_offset + in_size))) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
    }

    if (0 < in_size) {
        glapi.glBindBufferRange(util::gl_buffer_targets(in_target), 
                                in_index, _gl_buffer_id,
                                in_offset, in_size);
    }
    else {
        glapi.glBindBufferBase(util::gl_buffer_targets(in_target), 
                               in_index, _gl_buffer_id);
    }

    gl_assert(glapi, leaving buffer::bind_range());
}

void
buffer::unbind_range(render_context&   in_context,
                     buffer_binding    in_target,
                     const unsigned    in_index)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering buffer::unbind_range());

    assert(state().ok());

    glapi.glBindBufferBase(util::gl_buffer_targets(in_target), in_index, 0);

    gl_assert(glapi, leaving buffer::unbind_range());
}


void*
buffer::map(const render_context& in_context,
            const buffer_access   in_access)
{
    return (map_range(in_context, 0, _descriptor._size, in_access));
}

void*
buffer::map_range(const render_context& in_context,
                  scm::size_t           in_offset,
                  scm::size_t           in_size,
                  const buffer_access   in_access)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering buffer::map_range());

    assert(_gl_buffer_id != 0);
    assert(state().ok());

    void*       return_value = 0;
    unsigned    access_flags = util::gl_buffer_access(in_access);

    if (   (0 > in_offset)
        || (0 > in_size)
        || (_descriptor._size < (in_offset + in_size))
        || (0 == access_flags)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return (0);
    }

    if (   (0 != _mapped_interval_offset)
        || (0 != _mapped_interval_length)) {
        // buffer allready mapped
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return (0);
    }

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    return_value = glapi.glMapNamedBufferRangeEXT(_gl_buffer_id, in_offset, in_size, access_flags);

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    unsigned gl_buffer_target  = util::gl_buffer_targets(_descriptor._bindings);
    unsigned gl_buffer_binding = util::gl_buffer_bindings(_descriptor._bindings);

    util::buffer_binding_guard save_guard(glapi, gl_buffer_target, gl_buffer_binding);

    glapi.glBindBuffer(gl_buffer_target, _gl_buffer_id);
    return_value = glapi.glMapBufferRange(gl_buffer_target, in_offset, in_size, access_flags);

#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    if (0 != return_value) {
        _mapped_interval_offset = in_offset;
        _mapped_interval_length = in_size;
    }

    gl_assert(glapi, leaving buffer::map_range());

    return (return_value);
}

bool
buffer::unmap(const render_context& in_context)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering buffer::unmap());

    assert(_gl_buffer_id != 0);
    assert(state().ok());

    if (   0 == _mapped_interval_offset
        && 0 == _mapped_interval_length) {
        // buffer not mapped
        return (true);
    }

    bool return_value = true;
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    return_value = (GL_TRUE == glapi.glUnmapNamedBufferEXT(_gl_buffer_id));

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    unsigned gl_buffer_target  = util::gl_buffer_targets(_descriptor._bindings);
    unsigned gl_buffer_binding = util::gl_buffer_bindings(_descriptor._bindings);

    util::buffer_binding_guard save_guard(glapi, gl_buffer_target, gl_buffer_binding);

    glapi.glBindBuffer(gl_buffer_target, _gl_buffer_id);
    return_value = (GL_TRUE == glapi.glUnmapBuffer(gl_buffer_target));
        
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    _mapped_interval_offset = 0;
    _mapped_interval_length = 0;

    gl_assert(glapi, leaving buffer::unmap());

    return (return_value);
}

bool
buffer::buffer_data(      render_device&     ren_dev,
                    const descriptor_type&   buffer_desc,
                    const void*              initial_data)
{
    const opengl::gl3_core& glcore = ren_dev.opengl3_api();

    gl_assert(glcore, entering buffer::buffer_data());

    util::gl_error          glerror(glcore);

    if (0 == _gl_buffer_id) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return (false);
    }

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    glcore.glNamedBufferDataEXT(_gl_buffer_id,
                                buffer_desc._size,
                                initial_data,
                                util::gl_usage_flags(buffer_desc._usage));

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    unsigned gl_buffer_target  = util::gl_buffer_targets(buffer_desc._bindings);
    unsigned gl_buffer_binding = util::gl_buffer_bindings(buffer_desc._bindings);

    util::buffer_binding_guard save_guard(glcore, gl_buffer_target, gl_buffer_binding);

    glcore.glBindBuffer(gl_buffer_target, _gl_buffer_id);
    glcore.glBufferData(gl_buffer_target,
                        buffer_desc._size,
                        initial_data,
                        util::gl_usage_flags(buffer_desc._usage));

#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    if (glerror) {
        _descriptor = descriptor_type();
        state().set(glerror.to_object_state());
        return (false);
    }
    else {
        _descriptor = buffer_desc;
        return (true);
    }
}

bool
buffer::buffer_sub_data(render_device&  ren_dev,
                        scm::size_t     offset,
                        scm::size_t     size,
                        const void*     data)
{
    const opengl::gl3_core& glcore = ren_dev.opengl3_api();

    gl_assert(glcore, entering buffer::buffer_sub_data());

    util::gl_error          glerror(glcore);

    if (0 == _gl_buffer_id) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return (false);
    }

    if (0 > offset || 0 > size) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return (false);
    }

    if ((offset + size) > _descriptor._size) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return (false);
    }

    if (   offset < (_mapped_interval_offset + _mapped_interval_length)
        && (offset + size) > _mapped_interval_offset) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return (false);
    }

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    glcore.glNamedBufferSubDataEXT(_gl_buffer_id, offset, size, data);

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    unsigned gl_buffer_target  = util::gl_buffer_targets(_descriptor._bindings);
    unsigned gl_buffer_binding = util::gl_buffer_bindings(_descriptor._bindings);

    util::buffer_binding_guard save_guard(glcore, gl_buffer_target, gl_buffer_binding);

    glcore.glBindBuffer(gl_buffer_target, _gl_buffer_id);
    glcore.glBufferSubData(gl_buffer_target, offset, size, data);

#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    gl_assert(glcore, leaving buffer::buffer_sub_data());

    return (true);
}

const buffer::descriptor_type&
buffer::descriptor() const
{
    return (_descriptor);
}

void
buffer::print(std::ostream& os) const
{
    // TODO
}

} // namespace gl
} // namespace scm
