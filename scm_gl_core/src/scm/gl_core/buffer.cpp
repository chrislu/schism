
#include "buffer.h"

#include <cassert>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/opengl/config.h>
#include <scm/gl_core/opengl/gl3_core.h>
#include <scm/gl_core/opengl/util/assert.h>
#include <scm/gl_core/opengl/util/error_helper.h>

namespace  {

int gl_usage_flags[] = {
    // write once
    GL_STATIC_DRAW,     // GPU r,  CPU
    GL_STATIC_READ,     // GPU     CPU r
    GL_STATIC_COPY,     // GPU rw, CPU
    // low write frequency
    GL_STREAM_DRAW,     // GPU r,  CPU w
    GL_STREAM_READ,     // GPU w,  CPU r
    GL_STREAM_COPY,     // GPU rw, CPU
    // high write frequency
    GL_DYNAMIC_DRAW,    // GPU r,  CPU w
    GL_DYNAMIC_READ,    // GPU w,  CPU r
    GL_DYNAMIC_COPY     // GPU rw, CPU
};

} // namespace 
namespace scm {
namespace gl {

buffer::buffer(render_device& ren_dev,
               const descriptor_type&   buffer_desc,
               const void*              initial_data)
  : render_device_resource(ren_dev),
    _descriptor(),
    _gl_buffer_id(0),
    _mapped_interval_offset(0),
    _mapped_interval_length(0)
{
    const opengl::gl3_core& glcore = ren_dev.opengl3_api();

    glcore.glGenBuffers(1, &_gl_buffer_id);
    if (0 == _gl_buffer_id) {
        state().set(object_state::OS_BAD);
    }
    else {
        buffer_data(ren_dev, buffer_desc, initial_data);
    }
    gl_assert(glcore, leaving buffer::buffer());
}

buffer::~buffer()
{
    const opengl::gl3_core& glcore = parent_device().opengl3_api();

    assert(0 != _gl_buffer_id);
    glcore.glDeleteBuffers(1, &_gl_buffer_id);
    
    gl_assert(glcore, leaving buffer::~buffer());
}

bool
buffer::buffer_data(      render_device&     ren_dev,
                    const descriptor_type&   buffer_desc,
                    const void*              initial_data)
{
    const opengl::gl3_core& glcore = ren_dev.opengl3_api();
    util::gl_error          glerror(glcore);

    if (0 == _gl_buffer_id) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return (false);
    }

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    glcore.glNamedBufferDataEXT(_gl_buffer_id,
                                buffer_desc._size,
                                initial_data,
                                gl_usage_flags[buffer_desc._usage]);

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
#error "not implemented yet, and hopefully never will"
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
#error "not implemented yet, and hopefully never will"
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
