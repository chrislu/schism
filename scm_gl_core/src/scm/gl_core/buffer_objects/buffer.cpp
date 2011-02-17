
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

buffer::buffer(render_device&     ren_dev,
               const buffer_desc& in_desc,
               const void*        initial_data)
  : render_device_resource(ren_dev),
    _descriptor(),
    _mapped_interval_offset(0),
    _mapped_interval_length(0)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();

    glapi.glGenBuffers(1, &(context_bindable_object::_gl_object_id));
    if (0 == object_id()) {
        state().set(object_state::OS_BAD);
    }
    else {
        context_bindable_object::_gl_object_target  = util::gl_buffer_targets(in_desc._bindings);
        context_bindable_object::_gl_object_binding = util::gl_buffer_bindings(in_desc._bindings);
        buffer_data(ren_dev, in_desc, initial_data);
    }
    gl_assert(glapi, leaving buffer::buffer());
}

buffer::~buffer()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != object_id());
    glapi.glDeleteBuffers(1, &(context_bindable_object::_gl_object_id));
    
    gl_assert(glapi, leaving buffer::~buffer());
}

void
buffer::bind(render_context& ren_ctx, buffer_binding target) const
{
    const opengl::gl3_core& glapi = ren_ctx.opengl_api();

    gl_assert(glapi, entering buffer::bind());

    assert(object_id() != 0);
    assert(state().ok());

    glapi.glBindBuffer(util::gl_buffer_targets(target), object_id());

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

    assert(object_id() != 0);
    assert(state().ok());

    if (   (0 > in_offset)
        || (0 > in_size)
        || (_descriptor._size < (in_offset + in_size))) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
    }

    if (0 < in_size) {
        glapi.glBindBufferRange(util::gl_buffer_targets(in_target), 
                                in_index, object_id(),
                                in_offset, in_size);
    }
    else {
        glapi.glBindBufferBase(util::gl_buffer_targets(in_target), 
                               in_index, object_id());
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
            const access_mode   in_access)
{
    return (map_range(in_context, 0, _descriptor._size, in_access));
}

void*
buffer::map_range(const render_context& in_context,
                  scm::size_t           in_offset,
                  scm::size_t           in_size,
                  const access_mode   in_access)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering buffer::map_range());

    assert(object_id() != 0);
    assert(state().ok());

    void*       return_value = 0;
    unsigned    access_flags = util::gl_buffer_access_mode(in_access);

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

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        return_value = glapi.glMapNamedBufferRangeEXT(object_id(), in_offset, in_size, access_flags);
    }
    else {
        util::buffer_binding_guard save_guard(glapi, object_target(), object_binding());

        glapi.glBindBuffer(object_target(), object_id());
        return_value = glapi.glMapBufferRange(object_target(), in_offset, in_size, access_flags);
    }

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

    assert(object_id() != 0);
    assert(state().ok());

    if (   0 == _mapped_interval_offset
        && 0 == _mapped_interval_length) {
        // buffer not mapped
        return (true);
    }

    bool return_value = true;

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        return_value = (GL_TRUE == glapi.glUnmapNamedBufferEXT(object_id()));
    }
    else {
        util::buffer_binding_guard save_guard(glapi, object_target(), object_binding());

        glapi.glBindBuffer(object_target(), object_id());
        return_value = (GL_TRUE == glapi.glUnmapBuffer(object_target()));
    }

    _mapped_interval_offset = 0;
    _mapped_interval_length = 0;

    gl_assert(glapi, leaving buffer::unmap());

    return (return_value);
}

bool
buffer::buffer_data(const render_device& ren_dev,
                    const buffer_desc&   in_desc,
                    const void*          initial_data)
{
    const opengl::gl3_core& glcore = ren_dev.opengl3_api();

    gl_assert(glcore, entering buffer::buffer_data());

    util::gl_error          glerror(glcore);

    if (0 == object_id()) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return (false);
    }

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glcore.glNamedBufferDataEXT(object_id(),
                                    in_desc._size,
                                    initial_data,
                                    util::gl_usage_flags(in_desc._usage));
    }
    else {
        util::buffer_binding_guard save_guard(glcore, object_target(), object_binding());

        glcore.glBindBuffer(object_target(), object_id());
        glcore.glBufferData(object_target(),
                            in_desc._size,
                            initial_data,
                            util::gl_usage_flags(in_desc._usage));
    }

    if (glerror) {
        _descriptor = buffer_desc();
        state().set(glerror.to_object_state());
        return (false);
    }
    else {
        _descriptor = in_desc;
        return (true);
    }
}

bool
buffer::buffer_sub_data(const render_device& ren_dev,
                        scm::size_t          offset,
                        scm::size_t          size,
                        const void*          data)
{
    const opengl::gl3_core& glcore = ren_dev.opengl3_api();

    gl_assert(glcore, entering buffer::buffer_sub_data());

    util::gl_error          glerror(glcore);

    if (0 == object_id()) {
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

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glcore.glNamedBufferSubDataEXT(object_id(), offset, size, data);
    }
    else {
        util::buffer_binding_guard save_guard(glcore, object_target(), object_binding());

        glcore.glBindBuffer(object_target(), object_id());
        glcore.glBufferSubData(object_target(), offset, size, data);
    }

    gl_assert(glcore, leaving buffer::buffer_sub_data());

    return (true);
}

const buffer_desc&
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
