
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "frame_buffer.h"

#include <algorithm>
#include <cassert>
#include <limits>

#include <scm/gl_core/config.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/frame_buffer_objects/render_buffer.h>
#include <scm/gl_core/frame_buffer_objects/render_target.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

frame_buffer::attachment::attachment(const render_target_ptr& in_target, unsigned in_level, int in_layer, unsigned tex_target)
  : _target(in_target),
    _level(in_level),
    _layer(in_layer),
    _tex_target(tex_target)
{
}

bool
frame_buffer::attachment::operator==(const attachment& rhs) const
{
    return (   (_target     == rhs._target)
            && (_level      == rhs._level)
            && (_layer      == rhs._layer)
            && (_tex_target == rhs._tex_target));
}

bool
frame_buffer::attachment::operator!=(const attachment& rhs) const
{
    return (   (_target     != rhs._target)
            || (_level      != rhs._level)
            || (_layer      != rhs._layer)
            || (_tex_target != rhs._tex_target));
}

frame_buffer::frame_buffer(render_device& in_device)
  : render_device_child(in_device),
    _drawable_region((std::numeric_limits<unsigned>::max)()),
    _current_gl_binding(0),
    _attachments_dirty(true)
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    glapi.glGenFramebuffers(1, &(context_bindable_object::_gl_object_id));
    if (0 == object_id()) {
        state().set(object_state::OS_BAD);
    }
    else {
        context_bindable_object::_gl_object_target  = GL_DRAW_FRAMEBUFFER;
        context_bindable_object::_gl_object_binding = GL_DRAW_FRAMEBUFFER_BINDING;

        _selected_color_attachments.resize(in_device.capabilities()._max_frame_buffer_color_attachments);
        _current_color_attachments.resize(in_device.capabilities()._max_frame_buffer_color_attachments);
        _draw_buffers.resize(in_device.capabilities()._max_frame_buffer_color_attachments);
        std::fill(_draw_buffers.begin(), _draw_buffers.end(), GL_NONE);
    }

    gl_assert(glapi, leaving frame_buffer::frame_buffer());
}

frame_buffer::~frame_buffer()
{
    const opengl::gl_core& glapi = parent_device().opengl_api();

    assert(0 != object_id());
    glapi.glDeleteFramebuffers(1, &(context_bindable_object::_gl_object_id));
    
    gl_assert(glapi, leaving frame_buffer::~frame_buffer());
}

void
frame_buffer::attach_color_buffer(unsigned in_color_attachment, const render_target_ptr& in_target,
                                  unsigned in_level, unsigned in_layer, unsigned tex_target)
{
    assert(in_color_attachment < _selected_color_attachments.size());

    _selected_color_attachments[in_color_attachment] = attachment(in_target, in_level, in_layer, tex_target);

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
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());

    _current_gl_binding = util::gl_framebuffer_binding(in_binding);
    glapi.glBindFramebuffer(_current_gl_binding, object_id());

    gl_assert(glapi, leaving frame_buffer::bind());
}

void
frame_buffer::unbind(const render_context& in_context) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
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
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(in_buffer < _draw_buffers.size());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());

        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        glapi.glClearBufferfv(GL_COLOR, in_buffer, in_clear_color.data_array);
    }

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

void
frame_buffer::clear_color_buffer(const  render_context& in_context,
                                 const unsigned         in_buffer,
                                 const math::vec4i&     in_clear_color)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(in_buffer < _draw_buffers.size());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());

        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        glapi.glClearBufferiv(GL_COLOR, in_buffer, in_clear_color.data_array);
    }

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

void
frame_buffer::clear_color_buffer(const  render_context& in_context,
                                 const unsigned         in_buffer,
                                 const math::vec4ui&    in_clear_color)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(in_buffer < _draw_buffers.size());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());

        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        glapi.glClearBufferuiv(GL_COLOR, in_buffer, in_clear_color.data_array);
    }

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

void
frame_buffer::clear_color_buffers(const  render_context& in_context,
                                  const math::vec4f&     in_clear_color)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());
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
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    
    {
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());

        // apply attachments if they changed to get the correct attachment state for clearing
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        if (SCM_GL_CORE_USE_WORKAROUND_AMD) {
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

    gl_assert(glapi, leaving frame_buffer::clear_color_buffer());
}

void
frame_buffer::capture_color_buffer(      render_context& in_context,
                                   const unsigned        in_buffer,
                                   const texture_region& in_region,
                                   const data_format     in_data_format,
                                   const buffer_ptr&     in_target_buffer,
                                   const size_t          in_offset)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    
    {
        apply_attachments(in_context);
        assert(check_completeness(in_context));

        util::framebuffer_binding_guard fbo_guard(glapi, util::gl_framebuffer_binding(FRAMEBUFFER_READ),
                                                         util::gl_framebuffer_binding_point(FRAMEBUFFER_READ));
        //this->bind(in_context, FRAMEBUFFER_READ);
        glapi.glBindFramebuffer(util::gl_framebuffer_binding(FRAMEBUFFER_READ), object_id());

        util::buffer_binding_guard save_guard(glapi, util::gl_buffer_targets(BIND_PIXEL_PACK_BUFFER),
                                                     util::gl_buffer_bindings(BIND_PIXEL_PACK_BUFFER));

        glapi.glBindBuffer(GL_PIXEL_PACK_BUFFER, in_target_buffer->object_id());
        //in_target_buffer->bind(in_context, BIND_PIXEL_PACK_BUFFER);
    
        //glapi.glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        //glapi.glPixelStorei(GL_PACK_ALIGNMENT, 4);

        // TODO have the read buffer be part of the framebuffer state
        glapi.glReadBuffer(GL_COLOR_ATTACHMENT0 + in_buffer);
        gl_assert(glapi, frame_buffer::capture_color_buffer() after glReadBuffer to target attachment);

        glapi.glReadPixels(in_region._origin.x, in_region._origin.y,
                           in_region._dimensions.x, in_region._dimensions.y,
                           util::gl_base_format(in_data_format),
                           util::gl_base_type(in_data_format),
                           BUFFER_OFFSET(in_offset));

        gl_assert(glapi, frame_buffer::capture_color_buffer() after glReadPixels);

        glapi.glReadBuffer(GL_COLOR_ATTACHMENT0);
        gl_assert(glapi, frame_buffer::capture_color_buffer() after glReadBuffer to default attachment);
    }

    gl_assert(glapi, leaving frame_buffer::capture_color_buffer());

    // bind fbo (GUARDED)
    // bind PBO (GUARDED)
    // glReadBuffer(GL_COLOR_ATTACHMENT0 + in_buffer);
    // glReadPixels
}

bool
frame_buffer::check_completeness(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    unsigned status = 0;
    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        status = glapi.glCheckNamedFramebufferStatusEXT(object_id(), object_target());
    }
    else {
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());

        status = glapi.glCheckFramebufferStatus(object_target());
    }

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
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (_attachments_dirty) {

#if !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
        util::framebuffer_binding_guard fbo_guard(glapi, object_target(), object_binding());
        glapi.glBindFramebuffer(object_target(), object_id());
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
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                glapi.glFramebufferDrawBuffersEXT(object_id(), static_cast<int>(_draw_buffers.size()), &(_draw_buffers.front()));
            }
            else {
                glapi.glDrawBuffers(static_cast<int>(_draw_buffers.size()), &(_draw_buffers.front()));
            }
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
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (GL_RENDERBUFFER == in_attachment._target->object_target()) {
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glNamedFramebufferRenderbufferEXT(object_id(),
                                                    in_attach_point,
                                                    GL_RENDERBUFFER,
                                                    in_attachment._target->object_id());
        }
        else {
            glapi.glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                            in_attach_point,
                                            GL_RENDERBUFFER,
                                            in_attachment._target->object_id());
        }
        gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferRenderbuffer());
    }
    else {
        if (   (in_attachment._target->array_layers() > 1)
            && (in_attachment._layer >= 0)) {
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                glapi.glNamedFramebufferTextureLayerEXT(object_id(),
                                                        in_attach_point,
                                                        in_attachment._target->object_id(),
                                                        in_attachment._level,
                                                        in_attachment._layer);
            }
            else {
                glapi.glFramebufferTextureLayer(object_target(),
                                                in_attach_point,
                                                in_attachment._target->object_id(),
                                                in_attachment._level,
                                                in_attachment._layer);
            }
            gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferTextureLayer());
        }
        else {
            if (in_attachment._tex_target == 0) {
                if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                    glapi.glNamedFramebufferTextureEXT(object_id(),
                                                        in_attach_point,
                                                        in_attachment._target->object_id(),
                                                        in_attachment._level);
                }
                else {
                    glapi.glFramebufferTexture(object_target(),
                                                in_attach_point,
                                                in_attachment._target->object_id(),
                                                in_attachment._level);
                }
                gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferTexture());
            } 
            else {
                if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                    glapi.glNamedFramebufferTexture2DEXT(object_id(),
                                                        in_attach_point,
                                                        in_attachment._tex_target,
                                                        in_attachment._target->object_id(),
                                                        in_attachment._level);
                }
                else {
                    glapi.glFramebufferTexture2D(object_target(),
                                                in_attach_point,
                                                in_attachment._tex_target,
                                                in_attachment._target->object_id(),
                                                in_attachment._level);
                }
                gl_assert(glapi, frame_buffer::apply_attachment() after glFramebufferTexture2D());
            }
        }
    }
}

void
frame_buffer::clear_attachment(const render_context& in_context, unsigned in_attach_point)
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glapi.glNamedFramebufferRenderbufferEXT(object_id(), in_attach_point, GL_RENDERBUFFER, 0);
    }
    else {
        glapi.glFramebufferRenderbuffer(object_target(), in_attach_point, GL_RENDERBUFFER, 0);
    }

    gl_assert(glapi, frame_buffer::clear_attachment() after glFramebufferRenderbuffer());
}

void
frame_buffer::clear_color_attachments(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    std::fill(_selected_color_attachments.begin(), _selected_color_attachments.end(), attachment());
    _attachments_dirty = true;

    apply_attachments(in_context);

    gl_assert(glapi, frame_buffer::clear_attachments() after glFramebufferRenderbuffer());
}

void
frame_buffer::clear_depth_stencil_attachment(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    _selected_depth_stencil_attachment = attachment();
    _attachments_dirty = true;
    
    apply_attachments(in_context);

    gl_assert(glapi, frame_buffer::clear_attachments() after glFramebufferRenderbuffer());
}

void
frame_buffer::clear_attachments(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    std::fill(_selected_color_attachments.begin(), _selected_color_attachments.end(), attachment());
    _selected_depth_stencil_attachment = attachment();
    _attachments_dirty = true;
    
    apply_attachments(in_context);

    gl_assert(glapi, frame_buffer::clear_attachments() after glFramebufferRenderbuffer());
}

} // namespace gl
} // namespace scm
