
#include "rasterizer_state.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

// rasterizer_state ///////////////////////////////////////////////////////////////////////////////
rasterizer_state_desc::rasterizer_state_desc(
    fill_mode             in_fmode,
    cull_mode             in_cmode,
    polygon_orientation   in_fface,
    bool                  in_msample,
    bool                  in_sctest,
    bool                  in_smlines)
  : _fill_mode(in_fmode),
    _cull_mode(in_cmode),
    _front_face(in_fface),
    _multi_sample(in_msample),
    _scissor_test(in_sctest),
    _smooth_lines(in_smlines)
{
}

rasterizer_state::rasterizer_state(      render_device&         in_device,
                                   const rasterizer_state_desc& in_desc)
  : render_device_child(in_device),
    _descriptor(in_desc)
{
}

rasterizer_state::~rasterizer_state()
{
}

const rasterizer_state_desc&
rasterizer_state::descriptor() const
{
    return (_descriptor);
}

void
rasterizer_state::apply(const render_context&   in_context,
                        const rasterizer_state& in_applied_state) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    if (_descriptor._fill_mode != in_applied_state._descriptor._fill_mode) {
        glapi.glPolygonMode(GL_FRONT_AND_BACK, util::gl_fill_mode(_descriptor._fill_mode));
    }
    if (_descriptor._cull_mode != in_applied_state._descriptor._cull_mode) {
        if (_descriptor._cull_mode == CULL_NONE) {
            glapi.glDisable(GL_CULL_FACE);
        }
        else {
            glapi.glEnable(GL_CULL_FACE);
            glapi.glCullFace(util::gl_cull_mode(_descriptor._cull_mode));
        }
    }
    if (_descriptor._front_face != in_applied_state._descriptor._front_face) {
        glapi.glFrontFace(util::gl_polygon_orientation(_descriptor._front_face));
    }
    if (_descriptor._multi_sample != in_applied_state._descriptor._multi_sample) {
        if (_descriptor._multi_sample) {
            glapi.glEnable(GL_MULTISAMPLE);
        }
        else {
            glapi.glDisable(GL_MULTISAMPLE);
        }
    }
    if (_descriptor._scissor_test != in_applied_state._descriptor._scissor_test) {
        if (_descriptor._scissor_test) {
            glapi.glEnable(GL_SCISSOR_TEST);
        }
        else {
            glapi.glDisable(GL_SCISSOR_TEST);
        }
    }
    if (_descriptor._smooth_lines != in_applied_state._descriptor._smooth_lines) {
        if (_descriptor._smooth_lines) {
            glapi.glEnable(GL_LINE_SMOOTH);
        }
        else {
            glapi.glDisable(GL_LINE_SMOOTH);
        }
    }

    gl_assert(glapi, leaving rasterizer_state::apply());
}

void
rasterizer_state::force_apply(const render_context&   in_context) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    glapi.glPolygonMode(GL_FRONT_AND_BACK, util::gl_fill_mode(_descriptor._fill_mode));
    if (_descriptor._cull_mode == CULL_NONE) {
        glapi.glDisable(GL_CULL_FACE);
    }
    else {
        glapi.glEnable(GL_CULL_FACE);
        glapi.glCullFace(util::gl_cull_mode(_descriptor._cull_mode));
    }
    glapi.glFrontFace(util::gl_polygon_orientation(_descriptor._front_face));
    if (_descriptor._multi_sample) {
        glapi.glEnable(GL_MULTISAMPLE);
    }
    else {
        glapi.glDisable(GL_MULTISAMPLE);
    }
    if (_descriptor._scissor_test) {
        glapi.glEnable(GL_SCISSOR_TEST);
    }
    else {
        glapi.glDisable(GL_SCISSOR_TEST);
    }
    if (_descriptor._smooth_lines) {
        glapi.glEnable(GL_LINE_SMOOTH);
    }
    else {
        glapi.glDisable(GL_LINE_SMOOTH);
    }

    gl_assert(glapi, leaving rasterizer_state::force_apply());
}

} // namespace gl
} // namespace scm
