
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "blend_state.h"

#include <cassert>

#include <scm/core/math.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

// blend_state ////////////////////////////////////////////////////////////////////////////////////
blend_ops::blend_ops(bool            in_enabled,
                     blend_func      in_src_rgb_func,
                     blend_func      in_dst_rgb_func,
                     blend_func      in_src_alpha_func,
                     blend_func      in_dst_alpha_func,
                     blend_equation  in_rgb_equation,
                     blend_equation  in_alpha_equation,
                     unsigned        in_write_mask)
  : _enabled(in_enabled)
  , _src_rgb_func(in_src_rgb_func)
  , _dst_rgb_func(in_dst_rgb_func)
  , _src_alpha_func(in_src_alpha_func)
  , _dst_alpha_func(in_dst_alpha_func)
  , _rgb_equation(in_rgb_equation)
  , _alpha_equation(in_alpha_equation)
  , _write_mask(in_write_mask)
{
}

blend_ops_array
blend_ops::operator()(bool            in_enabled,
                      blend_func      in_src_rgb_func,
                      blend_func      in_dst_rgb_func,
                      blend_func      in_src_alpha_func,
                      blend_func      in_dst_alpha_func,
                      blend_equation  in_rgb_equation,
                      blend_equation  in_alpha_equation,
                      unsigned        in_write_mask)
{
    blend_ops_array ret(*this);

    return (ret(in_enabled,
                in_src_rgb_func,   in_dst_rgb_func,
                in_src_alpha_func, in_dst_alpha_func,
                in_rgb_equation,   in_alpha_equation,
                in_write_mask));
}

bool
blend_ops::operator==(const blend_ops& rhs) const
{
    return (   (_enabled == rhs._enabled)
            && (_src_rgb_func == rhs._src_rgb_func)
            && (_dst_rgb_func == rhs._dst_rgb_func)
            && (_rgb_equation == rhs._rgb_equation)
            && (_src_alpha_func == rhs._src_alpha_func)
            && (_dst_alpha_func == rhs._dst_alpha_func)
            && (_alpha_equation == rhs._alpha_equation)
            && (_write_mask == rhs._write_mask));
}

bool
blend_ops::operator!=(const blend_ops& rhs) const
{
    return (   (_enabled != rhs._enabled)
            || (_src_rgb_func != rhs._src_rgb_func)
            || (_dst_rgb_func != rhs._dst_rgb_func)
            || (_rgb_equation != rhs._rgb_equation)
            || (_src_alpha_func != rhs._src_alpha_func)
            || (_dst_alpha_func != rhs._dst_alpha_func)
            || (_alpha_equation != rhs._alpha_equation)
            || (_write_mask != rhs._write_mask));
}

blend_ops_array::blend_ops_array(const blend_ops_array& in_blend_op_array)
  : _array(in_blend_op_array._array)
{
}

blend_ops_array::blend_ops_array(const blend_ops& in_blend_ops)
  : _array(1, in_blend_ops)
{
}

blend_ops_array::blend_ops_array(bool            in_enabled,
                                 blend_func      in_src_rgb_func,
                                 blend_func      in_dst_rgb_func,
                                 blend_func      in_src_alpha_func,
                                 blend_func      in_dst_alpha_func,
                                 blend_equation  in_rgb_equation,
                                 blend_equation  in_alpha_equation,
                                 unsigned        in_write_mask)
  : _array(1, blend_ops(in_enabled,
                        in_src_rgb_func,   in_dst_rgb_func,
                        in_src_alpha_func, in_dst_alpha_func,
                        in_rgb_equation,   in_alpha_equation,
                        in_write_mask))
{
}

blend_ops_array&
blend_ops_array::operator()(const blend_ops& in_blend_ops)
{
    _array.push_back(in_blend_ops);
    return (*this);
}

blend_ops_array&
blend_ops_array::operator()(bool            in_enabled,
                            blend_func      in_src_rgb_func,
                            blend_func      in_dst_rgb_func,
                            blend_func      in_src_alpha_func,
                            blend_func      in_dst_alpha_func,
                            blend_equation  in_rgb_equation,
                            blend_equation  in_alpha_equation,
                            unsigned        in_write_mask)
{
    _array.push_back(blend_ops(in_enabled,
                               in_src_rgb_func,   in_dst_rgb_func,
                               in_src_alpha_func, in_dst_alpha_func,
                               in_rgb_equation,   in_alpha_equation,
                               in_write_mask));
    return (*this);
}

const blend_ops&
blend_ops_array::operator[](int in_index) const
{
    assert(in_index < _array.size());
    return (_array[in_index]);
}

size_t
blend_ops_array::size() const
{
    return (_array.size());
}

const blend_ops_array::blend_ops_vector&
blend_ops_array::blend_operations() const
{
    return (_array);
}

bool
blend_ops_array::operator==(const blend_ops_array& rhs) const
{
    return (_array == rhs._array);
}

bool
blend_ops_array::operator!=(const blend_ops_array& rhs) const
{
    return (_array != rhs._array);
}

blend_state_desc::blend_state_desc(const blend_ops& in_blend_ops,
                                   bool in_alpha_to_coverage)
  : _alpha_to_coverage(in_alpha_to_coverage)
  , _blend_ops(in_blend_ops)
{
    assert(1 == _blend_ops.size());
}

blend_state_desc::blend_state_desc(const blend_ops_array& in_blend_ops,
                                   bool in_alpha_to_coverage)
  : _blend_ops(in_blend_ops),
    _alpha_to_coverage(in_alpha_to_coverage)
{
    assert(in_blend_ops.size() == _blend_ops.size());
    assert(0 < _blend_ops.size());
}

blend_state::blend_state(      render_device&    in_device,
                         const blend_state_desc& in_desc)
  : render_device_child(in_device),
    _descriptor(in_desc)
{
    if (_descriptor._blend_ops.size() > in_device.capabilities()._max_draw_buffers) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
    }
}

blend_state::~blend_state()
{
}

const blend_state_desc&
blend_state::descriptor() const
{
    return (_descriptor);
}

void
blend_state::apply(const render_context& in_context, const math::vec4f& in_blend_color,
                   const blend_state&    in_applied_state, const math::vec4f& in_applied_blend_color) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (_descriptor._alpha_to_coverage != in_applied_state._descriptor._alpha_to_coverage) {
        if (_descriptor._alpha_to_coverage) {
            glapi.glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
        }
        else {
            glapi.glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
        }
    }

    if (in_blend_color != in_applied_blend_color) {
        glapi.glBlendColor(in_blend_color.r,
                           in_blend_color.g,
                           in_blend_color.b,
                           in_blend_color.a);
    }

    if (_descriptor._blend_ops.size() == 1) {
        if (in_applied_state._descriptor._blend_ops.size() == 1) {
            checked_apply(in_context, _descriptor._blend_ops[0], in_applied_state._descriptor._blend_ops[0]);
        }
        else {
            for (unsigned i = 0; i < in_applied_state._descriptor._blend_ops.size(); ++i) {
                force_disable_i(in_context, i);
            }
            force_apply(in_context, _descriptor._blend_ops[0]);
        }
    }
    else {
        if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
            glapi.glBlendEquationSeparate(util::gl_blend_equation(_descriptor._blend_ops[0]._rgb_equation),
                                          util::gl_blend_equation(_descriptor._blend_ops[0]._alpha_equation));
            glapi.glBlendFuncSeparate(util::gl_blend_func(_descriptor._blend_ops[0]._src_rgb_func),
                                      util::gl_blend_func(_descriptor._blend_ops[0]._dst_rgb_func),
                                      util::gl_blend_func(_descriptor._blend_ops[0]._src_alpha_func),
                                      util::gl_blend_func(_descriptor._blend_ops[0]._dst_alpha_func));
        }

        unsigned op_size = static_cast<unsigned>(_descriptor._blend_ops.size());
        unsigned ap_size = static_cast<unsigned>(in_applied_state._descriptor._blend_ops.size());
        unsigned max_size = math::max(op_size, ap_size);
        for (unsigned i = 0; i < max_size; ++i) {
            if ((i < op_size) && (i < ap_size)) {
                checked_apply_i(in_context, i, _descriptor._blend_ops[i], in_applied_state._descriptor._blend_ops[i]);
            }
            else if ((i < op_size) && !(i < ap_size)) {
                force_apply_i(in_context, i, _descriptor._blend_ops[i]);
            }
            else if (!(i < op_size) && (i < ap_size)) {
                force_disable_i(in_context, i);
            }
        }
    }

    gl_assert(glapi, leaving blend_state::apply());
}

void
blend_state::force_apply(const render_context& in_context, const math::vec4f& in_blend_color) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (_descriptor._alpha_to_coverage) {
        glapi.glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    }
    else {
        glapi.glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    }

    glapi.glBlendColor(in_blend_color.r,
                       in_blend_color.g,
                       in_blend_color.b,
                       in_blend_color.a);

    if (_descriptor._blend_ops.size() == 1) {
        force_apply(in_context, _descriptor._blend_ops[0]);
    }
    else {
        if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
            glapi.glBlendEquationSeparate(util::gl_blend_equation(_descriptor._blend_ops[0]._rgb_equation),
                                          util::gl_blend_equation(_descriptor._blend_ops[0]._alpha_equation));
            glapi.glBlendFuncSeparate(util::gl_blend_func(_descriptor._blend_ops[0]._src_rgb_func),
                                      util::gl_blend_func(_descriptor._blend_ops[0]._dst_rgb_func),
                                      util::gl_blend_func(_descriptor._blend_ops[0]._src_alpha_func),
                                      util::gl_blend_func(_descriptor._blend_ops[0]._dst_alpha_func));
        }

        for (int i = 0; i < _descriptor._blend_ops.size(); ++i) {
            force_apply_i(in_context, i, _descriptor._blend_ops[i]);
        }
    }

    gl_assert(glapi, leaving blend_state::force_apply());
}

void
blend_state::force_apply(const render_context& in_context,
                         const blend_ops&      in_blend_ops) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (in_blend_ops._enabled) {
        glapi.glEnable(GL_BLEND);
    }
    else {
        glapi.glDisable(GL_BLEND);
    }

    glapi.glColorMask(util::masked(in_blend_ops._write_mask, COLOR_RED),  util::masked(in_blend_ops._write_mask, COLOR_GREEN),
                      util::masked(in_blend_ops._write_mask, COLOR_BLUE), util::masked(in_blend_ops._write_mask, COLOR_ALPHA));

    glapi.glBlendEquationSeparate(util::gl_blend_equation(in_blend_ops._rgb_equation),
                                  util::gl_blend_equation(in_blend_ops._alpha_equation));
    glapi.glBlendFuncSeparate(util::gl_blend_func(in_blend_ops._src_rgb_func),   util::gl_blend_func(in_blend_ops._dst_rgb_func),
                              util::gl_blend_func(in_blend_ops._src_alpha_func), util::gl_blend_func(in_blend_ops._dst_alpha_func));

    gl_assert(glapi, leaving blend_state::force_apply());
}

void
blend_state::force_apply_i(const render_context& in_context,
                           unsigned              in_index,
                           const blend_ops&      in_blend_ops) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (in_blend_ops._enabled) {
        glapi.glEnablei(GL_BLEND, in_index);
    }
    else {
        glapi.glDisablei(GL_BLEND, in_index);
    }
    // color mask per render target
    glapi.glColorMaski(in_index, util::masked(in_blend_ops._write_mask, COLOR_RED),  util::masked(in_blend_ops._write_mask, COLOR_GREEN),
                                 util::masked(in_blend_ops._write_mask, COLOR_BLUE), util::masked(in_blend_ops._write_mask, COLOR_ALPHA));

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        glapi.glBlendEquationSeparatei(in_index, util::gl_blend_equation(in_blend_ops._rgb_equation),
                                                 util::gl_blend_equation(in_blend_ops._alpha_equation));
        glapi.glBlendFuncSeparatei(in_index, util::gl_blend_func(in_blend_ops._src_rgb_func),   util::gl_blend_func(in_blend_ops._dst_rgb_func),
                                             util::gl_blend_func(in_blend_ops._src_alpha_func), util::gl_blend_func(in_blend_ops._dst_alpha_func));
    }

    gl_assert(glapi, leaving blend_state::force_apply_i());
}

void
blend_state::checked_apply(const render_context& in_context,
                           const blend_ops&      in_blend_ops,
                           const blend_ops&      in_applied_blend_ops) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (in_blend_ops._enabled != in_applied_blend_ops._enabled) {
        if (in_blend_ops._enabled) {
            glapi.glEnable(GL_BLEND);
        }
        else {
            glapi.glDisable(GL_BLEND);
        }
    }
    if (in_blend_ops._write_mask != in_applied_blend_ops._write_mask) {
        glapi.glColorMask(util::masked(in_blend_ops._write_mask, COLOR_RED),  util::masked(in_blend_ops._write_mask, COLOR_GREEN),
                          util::masked(in_blend_ops._write_mask, COLOR_BLUE), util::masked(in_blend_ops._write_mask, COLOR_ALPHA));
    }
    glapi.glBlendEquationSeparate(util::gl_blend_equation(in_blend_ops._rgb_equation),
                                  util::gl_blend_equation(in_blend_ops._alpha_equation));
    glapi.glBlendFuncSeparate(util::gl_blend_func(in_blend_ops._src_rgb_func),   util::gl_blend_func(in_blend_ops._dst_rgb_func),
                              util::gl_blend_func(in_blend_ops._src_alpha_func), util::gl_blend_func(in_blend_ops._dst_alpha_func));

    gl_assert(glapi, leaving blend_state::checked_apply());
}

void
blend_state::checked_apply_i(const render_context& in_context,
                             unsigned              in_index,
                             const blend_ops&      in_blend_ops,
                             const blend_ops&      in_applied_blend_ops) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (in_blend_ops._enabled != in_applied_blend_ops._enabled) {
        if (in_blend_ops._enabled) {
            glapi.glEnablei(GL_BLEND, in_index);
        }
        else {
            glapi.glDisablei(GL_BLEND, in_index);
        }
    }
    if (in_blend_ops._write_mask != in_applied_blend_ops._write_mask) {
        glapi.glColorMaski(in_index, util::masked(in_blend_ops._write_mask, COLOR_RED),  util::masked(in_blend_ops._write_mask, COLOR_GREEN),
                                     util::masked(in_blend_ops._write_mask, COLOR_BLUE), util::masked(in_blend_ops._write_mask, COLOR_ALPHA));
    }
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        glapi.glBlendEquationSeparatei(in_index, util::gl_blend_equation(in_blend_ops._rgb_equation),
                                                 util::gl_blend_equation(in_blend_ops._alpha_equation));
        glapi.glBlendFuncSeparatei(in_index, util::gl_blend_func(in_blend_ops._src_rgb_func),   util::gl_blend_func(in_blend_ops._dst_rgb_func),
                                             util::gl_blend_func(in_blend_ops._src_alpha_func), util::gl_blend_func(in_blend_ops._dst_alpha_func));
    }
    gl_assert(glapi, leaving blend_state::checked_apply_i());
}

void
blend_state::force_disable_i(const render_context& in_context,
                             unsigned              in_index) const
{
    // apply default blend_ops to buffer index
    force_apply_i(in_context, in_index, blend_ops(false));
}

} // namespace gl
} // namespace scm
