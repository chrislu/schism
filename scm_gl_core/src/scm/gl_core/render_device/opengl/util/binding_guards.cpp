
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "binding_guards.h"

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

#include <scm/gl_core/config.h>

namespace scm {
namespace gl {

namespace util {

texture_binding_guard::texture_binding_guard(const opengl::gl_core& in_glapi,
                                             unsigned                in_target,
                                             unsigned                in_binding)
  : _save_active_texture_unit(0),
    _save_texture_object(0),
    _binding(in_binding),
    _target(in_target),
    _gl_api(in_glapi)
{
    gl_assert(_gl_api, entering texture_binding_guard::texture_binding_guard());

    //_gl_api.glGetIntegerv(GL_ACTIVE_TEXTURE, &_save_active_texture_unit);
    _gl_api.glGetIntegerv(_binding, &_save_texture_object);

    gl_assert(_gl_api, leaving texture_binding_guard::texture_binding_guard());
}

texture_binding_guard::~texture_binding_guard()
{
    gl_assert(_gl_api, entering texture_binding_guard::~texture_binding_guard());

    //_gl_api.glActiveTexture(_save_active_texture_unit);
    _gl_api.glBindTexture(_target, _save_texture_object);

    gl_assert(_gl_api, leaving texture_binding_guard::~texture_binding_guard());
}

program_binding_guard::program_binding_guard(const opengl::gl_core& in_glapi)
  : _gl_api(in_glapi)
{
    gl_assert(_gl_api, entering program_binding_guard::program_binding_guard());

    //if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_410) {
    //    _gl_api.glGetIntegerv(GL_ACTIVE_PROGRAM, &_active_program);
    //}
    //else {
        _gl_api.glGetIntegerv(GL_CURRENT_PROGRAM, &_active_program);
    //}

    gl_assert(_gl_api, leaving program_binding_guard::program_binding_guard());
}

program_binding_guard::~program_binding_guard()
{
    gl_assert(_gl_api, entering program_binding_guard::~program_binding_guard());

    _gl_api.glUseProgram(_active_program);

    gl_assert(_gl_api, leaving program_binding_guard::~program_binding_guard());
}

buffer_binding_guard::buffer_binding_guard(const opengl::gl_core& in_glapi,
                                           unsigned                in_target,
                                           unsigned                in_binding)
  : _binding(in_binding),
    _save(0),
    _target(in_target),
    _gl_api(in_glapi)
{
    gl_assert(_gl_api, entering buffer_binding_guard::buffer_binding_guard());

    _gl_api.glGetIntegerv(_binding, &_save);

    gl_assert(_gl_api, leaving buffer_binding_guard::buffer_binding_guard());
}

buffer_binding_guard::~buffer_binding_guard()
{
    gl_assert(_gl_api, entering buffer_binding_guard::~buffer_binding_guard());

    _gl_api.glBindBuffer(_target, _save);
    
    gl_assert(_gl_api, leaving buffer_binding_guard::~buffer_binding_guard());
}

vertex_array_binding_guard::vertex_array_binding_guard(const opengl::gl_core& in_glapi)
  : _gl_api(in_glapi),
    _save(0)
{
    gl_assert(_gl_api, entering vertex_array_binding_guard::vertex_array_binding_guard());

    _gl_api.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &_save);

    gl_assert(_gl_api, leaving vertex_array_binding_guard::vertex_array_binding_guard());
}

vertex_array_binding_guard::~vertex_array_binding_guard()
{
    gl_assert(_gl_api, entering vertex_array_binding_guard::~vertex_array_binding_guard());

    _gl_api.glBindVertexArray(_save);
    
    gl_assert(_gl_api, leaving vertex_array_binding_guard::~vertex_array_binding_guard());
}

framebuffer_binding_guard::framebuffer_binding_guard(const opengl::gl_core& in_glapi,
                                                     unsigned                in_target,
                                                     unsigned                in_binding)
  : _gl_api(in_glapi),
    _target(in_target),
    _binding(in_binding),
    _save(0)
{
    gl_assert(_gl_api, entering framebuffer_binding_guard::framebuffer_binding_guard());

    _gl_api.glGetIntegerv(_binding, &_save);
    
    gl_assert(_gl_api, leaving framebuffer_binding_guard::framebuffer_binding_guard());

}

framebuffer_binding_guard::~framebuffer_binding_guard()
{
    gl_assert(_gl_api, entering framebuffer_binding_guard::~framebuffer_binding_guard());

    _gl_api.glBindFramebuffer(_target, _save);
    
    gl_assert(_gl_api, leaving framebuffer_binding_guard::~framebuffer_binding_guard());

}

transform_feedback_binding_guard::transform_feedback_binding_guard(const opengl::gl_core& in_glapi,
                                                                   unsigned                in_target,
                                                                   unsigned                in_binding)
  : _gl_api(in_glapi)
  , _target(in_target)
  , _binding(in_binding)
  , _save(0)
{
    gl_assert(_gl_api, entering transform_feedback_binding_guard::transform_feedback_binding_guard());

    _gl_api.glGetIntegerv(_binding, &_save);
    
    gl_assert(_gl_api, leaving transform_feedback_binding_guard::transform_feedback_binding_guard());
}

transform_feedback_binding_guard::~transform_feedback_binding_guard()
{
    gl_assert(_gl_api, entering transform_feedback_binding_guard::~transform_feedback_binding_guard());

    _gl_api.glBindTransformFeedback(_target, _save);
    
    gl_assert(_gl_api, leaving transform_feedback_binding_guard::~transform_feedback_binding_guard());
}

} // namespace util
} // namespace gl
} // namespace scm
