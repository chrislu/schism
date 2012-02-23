
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_buffer.h"

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

texture_buffer_desc::texture_buffer_desc(const data_format   in_format,
                                         const buffer_ptr&   in_buffer)
  : _format(in_format)
  , _buffer(in_buffer)
{
}

bool
texture_buffer_desc::operator==(const texture_buffer_desc& rhs) const
{
    return (   (_format == rhs._format)
            && (_buffer == rhs._buffer));
}

bool
texture_buffer_desc::operator!=(const texture_buffer_desc& rhs) const
{
    return (   (_format != rhs._format)
            || (_buffer != rhs._buffer));
}

texture_buffer::texture_buffer(render_device&             in_device,
                               const texture_buffer_desc& in_desc)
  : texture(in_device)
  , _descriptor(in_desc)
{
    context_bindable_object::_gl_object_target  = GL_TEXTURE_BUFFER;
    context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_BUFFER;

    assert(state().ok());

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    if (SCM_GL_DEBUG) {
        // check if buffer size is less or equal to max texture buffer size and warn
        if (descriptor()._buffer->descriptor()._size > in_device.capabilities()._max_texture_buffer_size) {
            glerr() << log::warning << "texture_buffer::texture_buffer(): "
                    << "attaching buffer larger than supported texture buffer size "
                    << "(size: " << descriptor()._buffer->descriptor()._size
                    << ", max size: " <<in_device.capabilities()._max_texture_buffer_size << ")." << log::end;
        }
    }

#if !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
    glapi.glBindTexture(object_target(), object_id());
#endif // !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glapi.glTextureBufferEXT(object_id(),
                                 object_target(),
                                 util::gl_internal_format(descriptor()._format),
                                 descriptor()._buffer->object_id());
    }
    else {
        glapi.glTexBuffer(object_target(),
                          util::gl_internal_format(descriptor()._format),
                          descriptor()._buffer->object_id());
    }
    if (glerror) {
        state().set(glerror.to_object_state());
    }

    gl_assert(glapi, leaving texture_buffer::texture_buffer());
}

texture_buffer::~texture_buffer()
{
}

const texture_buffer_desc&
texture_buffer::descriptor() const
{
    return _descriptor;
}

} // namespace gl
} // namespace scm
