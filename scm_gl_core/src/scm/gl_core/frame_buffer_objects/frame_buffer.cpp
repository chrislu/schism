
#include "frame_buffer.h"

#include <algorithm>
#include <cassert>
#include <limits>

#include <scm/gl_core/config.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/frame_buffer_objects/render_buffer.h>
#include <scm/gl_core/frame_buffer_objects/render_target.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

frame_buffer::attachment::attachment(const render_target_ptr& in_target, unsigned in_level, int in_layer)
  : _target(in_target),
    _level(in_level),
    _layer(in_layer)
{
}

bool
frame_buffer::attachment::operator==(const attachment& rhs) const
{
    return (   (_target == rhs._target)
            && (_level  == rhs._level)
            && (_layer  == rhs._layer));
}

bool
frame_buffer::attachment::operator!=(const attachment& rhs) const
{
    return (   (_target != rhs._target)
            || (_level  != rhs._level)
            || (_layer  != rhs._layer));
}

frame_buffer::frame_buffer(render_device& in_device)
  : render_device_child(in_device),
    _gl_buffer_id(0),
    _drawable_region((std::numeric_limits<unsigned>::max)()),
    _current_gl_binding(0),
    _attachments_dirty(true)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();

    glapi.glGenFramebuffers(1, &_gl_buffer_id);
    if (0 == _gl_buffer_id) {
        state().set(object_state::OS_BAD);
    }
    else {
        _selected_color_attachments.resize(in_device.capabilities()._max_frame_buffer_color_attachments);
        _current_color_attachments.resize(in_device.capabilities()._max_frame_buffer_color_attachments);
        _draw_buffers.resize(in_device.capabilities()._max_frame_buffer_color_attachments);
        std::fill(_draw_buffers.begin(), _draw_buffers.end(), GL_NONE);
    }

    gl_assert(glapi, leaving frame_buffer::frame_buffer());
}

frame_buffer::~frame_buffer()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_buffer_id);
    glapi.glDeleteFramebuffers(1, &_gl_buffer_id);
    
    gl_assert(glapi, leaving frame_buffer::~frame_buffer());
}

unsigned
frame_buffer::buffer_id() const
{
    return (_gl_buffer_id);
}

void
frame_buffer::attach_color_buffer(unsigned in_color_attachment, const render_target_ptr& in_target,
                                  unsigned in_level, unsigned in_layer)
{
    assert(in_color_attachment < _selected_color_attachments.size());

    _selected_color_attachments[in_color_attachment] = attachment(in_target, in_level, in_layer);

    _drawable_region.x = math::min(in_target->dimensions().x, _drawable_region.x);
    _drawable_region.y = math::min(in_target->dimensions().y, _drawable_region.y);

    _attachments_dirty = true;
}

void
frame_buffer::attach_depth_stencil_buffer(const render_target_ptr& in_target,
                                          unsigned in_level, unsigned in_layer)
{
    if (   !is_depth_format(in_target->format())
        && !is_stencil_format(in_target->format())) {
        assert(0);
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return;
    }

    _selected_depth_stencil_attachment = attachment(in_target, in_level, in_layer);

    _drawable_region.x = math::min(in_target->dimensions().x, _drawable_region.x);
    _drawable_region.y = math::min(in_target->dimensions().y, _drawable_region.y);

    _attachments_dirty = true;
}

void
frame_buffer::clear_attachments()
{
    for (scm::size_t i = 0; i < _selected_color_attachments.size(); ++i) {
        _selected_color_attachments[i] = attachment();
        _draw_buffers[i]               = GL_NONE;
    }
    
    _selected_depth_stencil_attachment = attachment();

    _drawable_region = math::vec2ui((std::numeric_limits<unsigned>::max)());

    _attachments_dirty = true;
}

const math::vec2ui&
frame_buffer::drawable_region() const
{
    return (_drawable_region);
}

void
frame_buffer::bind(const render_context& in_context, frame_buffer_binding in_binding) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != buffer_id());

    _current_gl_binding = util::gl_framebuffer_binding(in_binding);
    glapi.glBindFramebuffer(_current_gl_binding, buffer_id());

    gl_assert(glapi, leaving frame_buffer::bind());
}

void
frame_buffer::unbind(const render_context& in_context) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != buffer_id());
    assert(0 != _current_gl_binding);

    if (0 != _current_gl_binding) {
        glapi.glBindFramebuffer(_current_gl_binding, 0);
        //glapi.glDrawBuffer(GL_BACK);
        _current_gl_binding = 0;
    }

    gl_assert(glapi, leaving frame_buffer::unbind());
}

void
frame_buffer::clear_color_buffer(const  render_context& in_context,
                                 const unsigned         in_buffer,
                                 const math::vec4f&     in_clear_color)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != buffer_id());
    assert(in_buffer < _draw_buffers.size());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING);
        glapi.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, buffer_id());

        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        glapi.glClearBufferfv(GL_COLOR, in_buffer, in_clear_color.data_array);
    }

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

void
frame_buffer::clear_color_buffers(const  render_context& in_context,
                                  const math::vec4f&     in_clear_color)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != buffer_id());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING);
        glapi.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, buffer_id());
        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        for (int i = 0; i < _draw_buffers.size(); ++i) {
            if (_draw_buffers[i] != GL_NONE){
                glapi.glClearBufferfv(GL_COLOR, i, in_clear_color.data_array);
            }
        }
    }

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

void
frame_buffer::clear_depth_stencil_buffer(const  render_context& in_context,
                                         const float            in_clear_depth,
                                         const int              in_clear_stencil)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != buffer_id());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING);
        glapi.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, buffer_id());

        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        //glapi.glClearDepth(in_clear_depth);
        //glapi.glClearStencil(in_clear_stencil);
        //glapi.glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        //glapi.glClearBufferfv(GL_DEPTH, 0, &in_clear_depth);
        glapi.glClearBufferfi(GL_DEPTH_STENCIL, 0, in_clear_depth, in_clear_stencil);
    }

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

bool
frame_buffer::check_completeness(const render_context& in_context)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    unsigned status = 0;
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    {
        status = glapi.glCheckNamedFramebufferStatusEXT(buffer_id(), GL_DRAW_FRAMEBUFFER);
    }
#else
    {
        util::framebuffer_binding_guard fbo_guard(glapi, GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING);
        glapi.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, buffer_id());

        status = glapi.glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    }
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    if (status != GL_FRAMEBUFFER_COMPLETE) {
        switch (status) {
        case GL_FRAMEBUFFER_UNDEFINED:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_UNDEFINED);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_INCOMPLETE_READ_BUFFER);
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_UNSUPPORTED);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            state().set(object_state::OS_ERROR_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS);
            break;
        default:
            state().set(object_state::OS_ERROR_UNKNOWN);
            break;
        }
        return (false);
    }
    else {
        return (true);
    }
}

void
frame_buffer::apply_attachments(const render_context& in_context)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    if (_attachments_dirty) {

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        util::framebuffer_binding_guard fbo_guard(glapi, GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING);
        glapi.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, buffer_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

        bool attachments_changed = false;
        for (unsigned i = 0; i < static_cast<unsigned>(_selected_color_attachments.size()); ++i) {
            const attachment& sca = _selected_color_attachments[i];
            attachment&       cca = _current_color_attachments[i];
            if (sca != cca) {
                if (sca._target) {
                    apply_attachment(in_context, GL_COLOR_ATTACHMENT0 + i, sca);
                    _draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i;
                    attachments_changed = true;
                }
                else {
                    if (cca._target) { // redundant, but for sanity sake
                        clear_attachment(in_context, GL_COLOR_ATTACHMENT0 + i);
                        _draw_buffers[i] = GL_NONE;
                        attachments_changed = true;
                    }
                    else {
                        assert(0);
                    }
                }
                cca = sca;
            }
        }
        if (true /*attachments_changed*/) {
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glFramebufferDrawBuffersEXT(buffer_id(), static_cast<int>(_draw_buffers.size()), &(_draw_buffers.front()));
#else
            glapi.glDrawBuffers(static_cast<int>(_draw_buffers.size()), &(_draw_buffers.front()));
#endif
        }
        if (_selected_depth_stencil_attachment != _current_depth_stencil_attachment) {

            if (_selected_depth_stencil_attachment._target) {
                unsigned attach_point = 0;

                if (is_depth_format(_selected_depth_stencil_attachment._target->format())) {
                    attach_point = GL_DEPTH_ATTACHMENT;
                }
                if (   is_depth_format(_selected_depth_stencil_attachment._target->format())
                    && is_stencil_format(_selected_depth_stencil_attachment._target->format())) {
                    attach_point = GL_DEPTH_STENCIL_ATTACHMENT;
                }

                if (0 == attach_point) {
                    //assert(0 != attach_point); // break in debug
                    state().set(object_state::OS_ERROR_INVALID_VALUE);
                    return;
                }

                apply_attachment(in_context, attach_point, _selected_depth_stencil_attachment);
            }
            else {
                if (_current_depth_stencil_attachment._target) { // redundant, but for sanity sake
                    clear_attachment(in_context, GL_DEPTH_STENCIL_ATTACHMENT);
                }
                else {
                    assert(0);
                }
            }
            _current_depth_stencil_attachment = _selected_depth_stencil_attachment;
        }
        _attachments_dirty = false;
    }
    
    gl_assert(glapi, leaving frame_buffer::apply_attachments());
}

void
frame_buffer::apply_attachment(const render_context& in_context, unsigned in_attach_point, const attachment& in_attachment)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    if (GL_RENDERBUFFER == in_attachment._target->object_target()) {
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        glapi.glNamedFramebufferRenderbufferEXT(buffer_id(),
                                                in_attach_point,
                                                GL_RENDERBUFFER,
                                                in_attachment._target->object_id());
#else
        glapi.glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                        in_attach_point,
                                        GL_RENDERBUFFER,
                                        in_attachment._target->object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferRenderbuffer());
    }
    else {
        if (   (in_attachment._target->array_layers() > 1)
            && (in_attachment._layer >= 0)) {
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                glapi.glNamedFramebufferTextureLayerEXT(buffer_id(),
                                                        in_attach_point,
                                                        in_attachment._target->object_id(),
                                                        in_attachment._level,
                                                        in_attachment._layer);
#else
                glapi.glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER,
                                                in_attach_point,
                                                in_attachment._target->object_id(),
                                                in_attachment._level,
                                                in_attachment._layer);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferTextureLayer());
        }
        else {
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glNamedFramebufferTextureEXT(buffer_id(),
                                                in_attach_point,
                                                in_attachment._target->object_id(),
                                                in_attachment._level);
#else
            glapi.glFramebufferTexture(GL_DRAW_FRAMEBUFFER,
                                        in_attach_point,
                                        in_attachment._target->object_id(),
                                        in_attachment._level);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferTexture());
        }
    }
}

void
frame_buffer::clear_attachment(const render_context& in_context, unsigned in_attach_point)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    glapi.glNamedFramebufferRenderbufferEXT(buffer_id(), in_attach_point, GL_RENDERBUFFER, 0);
#else
    glapi.glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, in_attach_point, GL_RENDERBUFFER, 0);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    gl_assert(glapi, frame_buffer::clear_attachment() after glFramebufferRenderbuffer());
}

} // namespace gl
} // namespace scm
