
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "depth_stencil_state.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

// depth_stencil_state ////////////////////////////////////////////////////////////////////////////

stencil_ops::stencil_ops(compare_func in_stencil_func,
                         stencil_op   in_stencil_sfail,
                         stencil_op   in_stencil_dfail,
                         stencil_op   in_stencil_dpass)
  : _stencil_func(in_stencil_func)
  , _stencil_sfail(in_stencil_sfail)
  , _stencil_dfail(in_stencil_dfail)
  , _stencil_dpass(in_stencil_dpass)
{
}

bool
stencil_ops::operator==(const stencil_ops& rhs) const
{
    return (   (_stencil_func  == rhs._stencil_func)
            && (_stencil_sfail == rhs._stencil_sfail)
            && (_stencil_dfail == rhs._stencil_dfail)
            && (_stencil_dpass == rhs._stencil_dpass));
}

bool
stencil_ops::operator!=(const stencil_ops& rhs) const
{
    return (   (_stencil_func  != rhs._stencil_func)
            || (_stencil_sfail != rhs._stencil_sfail)
            || (_stencil_dfail != rhs._stencil_dfail)
            || (_stencil_dpass != rhs._stencil_dpass));
}

depth_stencil_state_desc::depth_stencil_state_desc(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                                   bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                                   const stencil_ops& in_stencil_ops)
  : _depth_test(in_depth_test)
  , _depth_mask(in_depth_mask)
  , _depth_func(in_depth_func)
  , _stencil_test(in_stencil_test)
  , _stencil_rmask(in_stencil_rmask)
  , _stencil_wmask(in_stencil_wmask)
  , _stencil_front_ops(in_stencil_ops)
  , _stencil_back_ops(in_stencil_ops)
{
}

depth_stencil_state_desc::depth_stencil_state_desc(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                                   bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                                   const stencil_ops& in_stencil_front_ops, const stencil_ops& in_stencil_back_ops)
  : _depth_test(in_depth_test)
  , _depth_mask(in_depth_mask)
  , _depth_func(in_depth_func)
  , _stencil_test(in_stencil_test)
  , _stencil_rmask(in_stencil_rmask)
  , _stencil_wmask(in_stencil_wmask)
  , _stencil_front_ops(in_stencil_front_ops)
  , _stencil_back_ops(in_stencil_back_ops)
{
}

depth_stencil_state::depth_stencil_state(render_device&                  in_device,
                                         const depth_stencil_state_desc& in_desc)
  : render_device_child(in_device),
    _descriptor(in_desc)
{
}

depth_stencil_state::~depth_stencil_state()
{
}

const depth_stencil_state_desc&
depth_stencil_state::descriptor() const
{
    return (_descriptor);
}

void
depth_stencil_state::apply(const render_context& in_context, unsigned in_stencil_ref,
                           const depth_stencil_state& in_applied_state, unsigned in_applied_stencil_ref) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    // depth state
    if (_descriptor._depth_test != in_applied_state._descriptor._depth_test) {
        if (_descriptor._depth_test) {
            glapi.glEnable(GL_DEPTH_TEST);
        }
        else {
            glapi.glDisable(GL_DEPTH_TEST);
        }
    }
    if (_descriptor._depth_mask != in_applied_state._descriptor._depth_mask) {
        glapi.glDepthMask(_descriptor._depth_mask ? GL_TRUE : GL_FALSE);
    }
    if (_descriptor._depth_func != in_applied_state._descriptor._depth_func) {
        glapi.glDepthFunc(util::gl_compare_func(_descriptor._depth_func));
    }
    gl_assert(glapi, depth_stencil_state::apply(): after applying depth state);

    // stencil state
    if (_descriptor._stencil_test != in_applied_state._descriptor._stencil_test) {
        if (_descriptor._stencil_test) {
            glapi.glEnable(GL_STENCIL_TEST);
        }
        else {
            glapi.glDisable(GL_STENCIL_TEST);
        }
    }
    if (_descriptor._stencil_wmask != in_applied_state._descriptor._stencil_wmask) {
        glapi.glStencilMask(_descriptor._stencil_wmask);
    }
    if (   (_descriptor._stencil_front_ops != in_applied_state._descriptor._stencil_front_ops)
        || (_descriptor._stencil_rmask     != in_applied_state._descriptor._stencil_rmask)
        || (in_stencil_ref != in_applied_stencil_ref)) {
        glapi.glStencilFuncSeparate(GL_FRONT,
                                    util::gl_compare_func(_descriptor._stencil_front_ops._stencil_func),
                                    in_stencil_ref,
                                    _descriptor._stencil_rmask);
        glapi.glStencilOpSeparate(GL_FRONT,
                                  util::gl_stencil_op(_descriptor._stencil_front_ops._stencil_sfail),
                                  util::gl_stencil_op(_descriptor._stencil_front_ops._stencil_dfail),
                                  util::gl_stencil_op(_descriptor._stencil_front_ops._stencil_dpass));
        gl_assert(glapi, depth_stencil_state::apply(): after applying front stencil state);
    }
    if (   (_descriptor._stencil_back_ops  != in_applied_state._descriptor._stencil_back_ops)
        || (_descriptor._stencil_rmask     != in_applied_state._descriptor._stencil_rmask)
        || (in_stencil_ref != in_applied_stencil_ref)) {
        glapi.glStencilFuncSeparate(GL_BACK,
                                    util::gl_compare_func(_descriptor._stencil_back_ops._stencil_func),
                                    in_stencil_ref,
                                    _descriptor._stencil_rmask);
        glapi.glStencilOpSeparate(GL_BACK,
                                  util::gl_stencil_op(_descriptor._stencil_back_ops._stencil_sfail),
                                  util::gl_stencil_op(_descriptor._stencil_back_ops._stencil_dfail),
                                  util::gl_stencil_op(_descriptor._stencil_back_ops._stencil_dpass));
        gl_assert(glapi, depth_stencil_state::apply(): after applying back stencil state);
    }

    gl_assert(glapi, leaving depth_stencil_state::apply());
}

void
depth_stencil_state::force_apply(const render_context& in_context, unsigned in_stencil_ref) const
{
    const opengl::gl_core&         glapi         = in_context.opengl_api();

    // depth state
    if (_descriptor._depth_test) {
        glapi.glEnable(GL_DEPTH_TEST);
    }
    else {
        glapi.glDisable(GL_DEPTH_TEST);
    }
    glapi.glDepthMask(_descriptor._depth_mask ? GL_TRUE : GL_FALSE);
    glapi.glDepthFunc(util::gl_compare_func(_descriptor._depth_func));

    gl_assert(glapi, depth_stencil_state::force_apply(): after applying depth state);

    // stencil state
    if (_descriptor._stencil_test) {
        glapi.glEnable(GL_STENCIL_TEST);
    }
    else {
        glapi.glDisable(GL_STENCIL_TEST);
    }
    glapi.glStencilMask(_descriptor._stencil_wmask);
    glapi.glStencilFuncSeparate(GL_FRONT,
                                util::gl_compare_func(_descriptor._stencil_front_ops._stencil_func),
                                in_stencil_ref,
                                _descriptor._stencil_rmask);
    glapi.glStencilOpSeparate(GL_FRONT,
                              util::gl_stencil_op(_descriptor._stencil_front_ops._stencil_sfail),
                              util::gl_stencil_op(_descriptor._stencil_front_ops._stencil_dfail),
                              util::gl_stencil_op(_descriptor._stencil_front_ops._stencil_dpass));
    gl_assert(glapi, depth_stencil_state::force_apply(): after applying front stencil state);

    glapi.glStencilFuncSeparate(GL_BACK,
                                util::gl_compare_func(_descriptor._stencil_back_ops._stencil_func),
                                in_stencil_ref,
                                _descriptor._stencil_rmask);
    glapi.glStencilOpSeparate(GL_BACK,
                              util::gl_stencil_op(_descriptor._stencil_back_ops._stencil_sfail),
                              util::gl_stencil_op(_descriptor._stencil_back_ops._stencil_dfail),
                              util::gl_stencil_op(_descriptor._stencil_back_ops._stencil_dpass));
    gl_assert(glapi, depth_stencil_state::force_apply(): after applying back stencil state);

    gl_assert(glapi, leaving depth_stencil_state::force_apply());
}


} // namespace gl
} // namespace scm
