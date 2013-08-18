
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_CONTEXT_GUARDS_H_INCLUDED
#define SCM_GL_CORE_CONTEXT_GUARDS_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/math.h>

#include <scm/gl_core/render_device/context.h>

namespace scm {
namespace gl {

class context_program_guard : boost::noncopyable
{
public:
    context_program_guard(const render_context_ptr& in_context)
        : _guarded_context(in_context)
        , _save_program(in_context->current_program())
    {}
    ~context_program_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->bind_program(_save_program);
    }
private:
    const render_context_ptr& _guarded_context;
    const program_ptr         _save_program;
}; // class context_program_guard


class context_vertex_input_guard : boost::noncopyable
{
public:
    context_vertex_input_guard(const render_context_ptr& in_context)
        : _guarded_context(in_context)
        ,  _save_vertex_array(in_context->current_vertex_array())
        ,  _save_index_buffer_binding(in_context->current_index_buffer_binding())
    {
    }
    ~context_vertex_input_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->bind_vertex_array(_save_vertex_array);
        _guarded_context->set_index_buffer_binding(_save_index_buffer_binding);
    }
private:
    const render_context_ptr&   _guarded_context;
    const vertex_array_ptr      _save_vertex_array;
    const render_context::index_buffer_binding _save_index_buffer_binding;
}; // class context_vertex_input_guard


class context_uniform_buffer_guard : boost::noncopyable
{
public:
    context_uniform_buffer_guard(const render_context_ptr& in_context)
        : _guarded_context(in_context)
        , _save_uniform_buffers(in_context->current_uniform_buffers())
    {
    }
    ~context_uniform_buffer_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->set_uniform_buffers(_save_uniform_buffers);
    }
private:
    const render_context_ptr&                   _guarded_context;
    const render_context::buffer_binding_array  _save_uniform_buffers;
}; // class context_uniform_buffer_guard

class context_atomic_counter_buffer_guard : boost::noncopyable
{
public:
    context_atomic_counter_buffer_guard(const render_context_ptr& in_context)
        : _guarded_context(in_context)
        , _save_atomic_counter_buffers(in_context->current_atomic_counter_buffers())
    {
    }
    ~context_atomic_counter_buffer_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->set_atomic_counter_buffers(_save_atomic_counter_buffers);
    }
private:
    const render_context_ptr&                   _guarded_context;
    const render_context::buffer_binding_array  _save_atomic_counter_buffers;
}; // class context_atomic_counter_buffer_guard

class context_storage_buffer_guard : boost::noncopyable
{
public:
    context_storage_buffer_guard(const render_context_ptr& in_context)
        : _guarded_context(in_context)
        , _save_storage_buffers(in_context->current_storage_buffers())
    {
    }
    ~context_storage_buffer_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->set_storage_buffers(_save_storage_buffers);
    }
private:
    const render_context_ptr&                   _guarded_context;
    const render_context::buffer_binding_array  _save_storage_buffers;
}; // class context_storage_buffer_guard

class context_unpack_buffer_guard : boost::noncopyable
{
public:
    context_unpack_buffer_guard(const render_context_ptr& in_context)
        : _guarded_context(in_context)
        , _save_unpack_buffer(in_context->current_unpack_buffer())
    {
    }
    ~context_unpack_buffer_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->bind_unpack_buffer(_save_unpack_buffer);
    }
private:
    const render_context_ptr&   _guarded_context;
    const buffer_ptr            _save_unpack_buffer;
}; // class context_unpack_buffer_guard

class context_state_objects_guard : boost::noncopyable
{
public:
    context_state_objects_guard(const render_context_ptr& in_context)
      : _guarded_context(in_context)
      , _save_depth_stencil_state(in_context->current_depth_stencil_state())
      , _save_stencil_ref(in_context->current_stencil_ref_value())
      , _save_rasterizer_state(in_context->current_rasterizer_state())
      , _save_line_width(in_context->current_line_width())
      , _save_point_size(in_context->current_point_size())
      , _save_blend_state(in_context->current_blend_state())
      , _save_blend_color(in_context->current_blend_color())
    {
    }
    ~context_state_objects_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->set_depth_stencil_state(_save_depth_stencil_state, _save_stencil_ref);
        _guarded_context->set_rasterizer_state(_save_rasterizer_state, _save_line_width, _save_point_size);
        _guarded_context->set_blend_state(_save_blend_state, _save_blend_color);
    }
private:
    const render_context_ptr&       _guarded_context;
    const depth_stencil_state_ptr   _save_depth_stencil_state;
    const unsigned                  _save_stencil_ref;
    const rasterizer_state_ptr      _save_rasterizer_state;
    const float                     _save_line_width;
    const float                     _save_point_size;
    const blend_state_ptr           _save_blend_state;
    const math::vec4f               _save_blend_color;
}; // class context_state_objects_guard


class context_texture_units_guard : boost::noncopyable
{
public:
    context_texture_units_guard(const render_context_ptr& in_context)
      : _guarded_context(in_context)
      , _save_texture_units(in_context->current_texture_unit_state())
    {
    }
    ~context_texture_units_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->set_texture_unit_state(_save_texture_units);
    }
private:
    const render_context_ptr&                 _guarded_context;
    const render_context::texture_unit_array  _save_texture_units;
}; // class context_texture_units_guard

class context_image_units_guard : boost::noncopyable
{
public:
    context_image_units_guard(const render_context_ptr& in_context)
      : _guarded_context(in_context)
      , _save_image_units(in_context->current_image_unit_state())
    {
    }
    ~context_image_units_guard()
    {
        restore();
    }
    void restore()
    {
        _guarded_context->set_image_unit_state(_save_image_units);
    }
private:
    const render_context_ptr&               _guarded_context;
    const render_context::image_unit_array  _save_image_units;
}; // class context_image_units_guard


class context_framebuffer_guard : boost::noncopyable
{
public:
    context_framebuffer_guard(const render_context_ptr& in_context)
      : _guarded_context(in_context)
      , _save_framebuffer(in_context->current_frame_buffer())
      , _save_default_framebuffer_target(in_context->current_default_frame_buffer_target())
      , _save_viewports(in_context->current_viewports())
    {
    }
    ~context_framebuffer_guard()
    {
        restore();
    }
    void restore()
    {
        if (_save_framebuffer) {
            _guarded_context->set_frame_buffer(_save_framebuffer);
        }
        else {
            _guarded_context->set_default_frame_buffer(_save_default_framebuffer_target);
        }
        _guarded_context->set_viewports(_save_viewports);
    }
private:
    const render_context_ptr&               _guarded_context;
    const frame_buffer_ptr                  _save_framebuffer;
    const frame_buffer_target               _save_default_framebuffer_target;
    const viewport_array                    _save_viewports;
}; // class context_framebuffer_guard


class context_all_guard : boost::noncopyable
{
public:
    context_all_guard(const render_context_ptr& in_context)
      : _p_guard(in_context)
      , _v_guard(in_context)
      , _u_guard(in_context)
      , _ac_guard(in_context)
      , _ss_guard(in_context)
      , _up_guard(in_context)
      , _s_guard(in_context)
      , _t_guard(in_context)
      , _i_guard(in_context)
      , _f_guard(in_context)
    {
    }
    ~context_all_guard()
    {
    }
private:
    context_program_guard               _p_guard;
    context_vertex_input_guard          _v_guard;
    context_uniform_buffer_guard        _u_guard;
    context_atomic_counter_buffer_guard _ac_guard;
    context_storage_buffer_guard        _ss_guard;
    context_unpack_buffer_guard         _up_guard;
    context_state_objects_guard         _s_guard;
    context_texture_units_guard         _t_guard;
    context_image_units_guard           _i_guard;
    context_framebuffer_guard           _f_guard;
}; // class context_all_guard

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_CONTEXT_GUARDS_H_INCLUDED
