

#include "render_buffer.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

render_buffer_desc::render_buffer_desc(const math::vec2ui& in_size,
                                       const data_format   in_format,
                                       const unsigned      in_samples)
  : _size(in_size),
    _format(in_format),
    _samples(in_samples)
{
}

bool
render_buffer_desc::operator==(const render_buffer_desc& rhs) const
{
    return (   (_size         == rhs._size)
            && (_format       == rhs._format)
            && (_samples      == rhs._samples));
}

bool
render_buffer_desc::operator!=(const render_buffer_desc& rhs) const
{
    return (   (_size         != rhs._size)
            || (_format       != rhs._format)
            || (_samples      != rhs._samples));
}

render_buffer::render_buffer(render_device&            in_device,
                             const render_buffer_desc& in_desc)
  : render_target(in_device),
    _descriptor(in_desc)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();
    util::gl_error          glerror(glapi);

    glapi.glGenRenderbuffers(1, &_gl_object_id);
    if (0 == _gl_object_id) {
        state().set(object_state::OS_BAD);
    }
    else {

        _gl_object_target = GL_RENDERBUFFER;

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        glapi.glBindRenderbuffer(object_target(), object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        if (descriptor()._samples == 1) {
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glNamedRenderbufferStorageEXT(object_id(),
                                                util::gl_internal_format(descriptor()._format),
                                                descriptor()._size.x, descriptor()._size.y);
#else
            glapi.glRenderbufferStorage(object_target(),
                                        util::gl_internal_format(descriptor()._format),
                                        descriptor()._size.x, descriptor()._size.y);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            if (glerror) {
                state().set(glerror.to_object_state());
            }
        }
        else if (descriptor()._samples > 1) {
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glNamedRenderbufferStorageMultisampleEXT(object_id(),
                                        descriptor()._samples,
                                        util::gl_internal_format(descriptor()._format),
                                        descriptor()._size.x, descriptor()._size.y);
#else
            glapi.glRenderbufferStorageMultisample(object_target(),
                                        descriptor()._samples,
                                        util::gl_internal_format(descriptor()._format),
                                        descriptor()._size.x, descriptor()._size.y);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            if (glerror) {
                state().set(glerror.to_object_state());
            }
        }
        else {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
        }
#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        glapi.glBindRenderbuffer(object_target(), 0);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    }

    gl_assert(glapi, leaving render_buffer::render_buffer());
}

render_buffer::~render_buffer()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_object_id);
    glapi.glDeleteRenderbuffers(1, &_gl_object_id);
    
    gl_assert(glapi, leaving render_buffer::~render_buffer());
}

const render_buffer_desc&
render_buffer::descriptor() const
{
    return (_descriptor);
}

data_format
render_buffer::format() const
{
    return (_descriptor._format);
}

math::vec2ui
render_buffer::dimensions() const
{
    return (_descriptor._size);
}

unsigned
render_buffer::array_layers() const
{
    return (1);
}

unsigned
render_buffer::mip_map_layers() const
{
    return (1);
}

unsigned
render_buffer::samples() const
{
    return (_descriptor._samples);
}

} // namespace gl
} // namespace scm
