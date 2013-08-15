
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "rasterizer_state.h"

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

// rasterizer_state ///////////////////////////////////////////////////////////////////////////////
point_raster_state::point_raster_state(bool            in_shader_point_size,
                                       origin_mode     in_point_origin,
                                       float           in_point_fade_threshold)
  : _shader_point_size(in_shader_point_size)
  , _point_origin_mode(in_point_origin)
  , _point_fade_threshold(in_point_fade_threshold)
{
}

bool
point_raster_state::operator==(const point_raster_state& rhs) const
{
    return (   (_shader_point_size    == rhs._shader_point_size)
            && (_point_origin_mode    == rhs._point_origin_mode)
            && (_point_fade_threshold == rhs._point_fade_threshold));
}

bool
point_raster_state::operator!=(const point_raster_state& rhs) const
{
    return (   (_shader_point_size    != rhs._shader_point_size)
            || (_point_origin_mode    != rhs._point_origin_mode)
            || (_point_fade_threshold != rhs._point_fade_threshold));
}

rasterizer_state_desc::rasterizer_state_desc(
    fill_mode                 in_fmode,
    cull_mode                 in_cmode,
    polygon_orientation       in_fface,
    bool                      in_msample,
    bool                      in_sshading,
    float32                   in_min_sshading,
    bool                      in_sctest,
    bool                      in_smlines,
    const point_raster_state& in_point_state)
  : _fill_mode(in_fmode)
  , _cull_mode(in_cmode)
  , _front_face(in_fface)
  , _multi_sample(in_msample)
  , _sample_shading(in_sshading)
  , _min_sample_shading(in_min_sshading)
  , _scissor_test(in_sctest)
  , _smooth_lines(in_smlines)
  , _point_state(in_point_state)
{
}

rasterizer_state::rasterizer_state(      render_device&         in_device,
                                   const rasterizer_state_desc& in_desc)
  : render_device_child(in_device)
  , _descriptor(in_desc)
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
                        const float             in_line_width,
                        const float             in_point_size,
                        const rasterizer_state& in_applied_state,
                        const float             in_applied_line_width,
                        const float             in_applied_point_size) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    if (_descriptor._fill_mode != in_applied_state._descriptor._fill_mode) {
        glapi.glPolygonMode(GL_FRONT_AND_BACK, util::gl_fill_mode(_descriptor._fill_mode));
        gl_assert(glapi, rasterizer_state::apply() after glPolygonMode);
    }
    if (_descriptor._cull_mode != in_applied_state._descriptor._cull_mode) {
        if (_descriptor._cull_mode == CULL_NONE) {
            glapi.glDisable(GL_CULL_FACE);
        }
        else {
            glapi.glEnable(GL_CULL_FACE);
            glapi.glCullFace(util::gl_cull_mode(_descriptor._cull_mode));
        }
        gl_assert(glapi, rasterizer_state::apply() after glCullFace);
    }
    if (_descriptor._front_face != in_applied_state._descriptor._front_face) {
        glapi.glFrontFace(util::gl_polygon_orientation(_descriptor._front_face));
        gl_assert(glapi, rasterizer_state::apply() after glFrontFace);
    }
    if (_descriptor._multi_sample != in_applied_state._descriptor._multi_sample) {
        if (_descriptor._multi_sample) {
            glapi.glEnable(GL_MULTISAMPLE);
        }
        else {
            glapi.glDisable(GL_MULTISAMPLE);
        }
        gl_assert(glapi, rasterizer_state::apply() after GL_MULTISAMPLE);
    }

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        if (   _descriptor._multi_sample
            && _descriptor._sample_shading)
        {
            glapi.glEnable(GL_SAMPLE_SHADING);
        }
        else {
            glapi.glDisable(GL_SAMPLE_SHADING);
        }
        gl_assert(glapi, rasterizer_state::apply() after GL_SAMPLE_SHADING);

        glapi.glMinSampleShading(math::clamp(_descriptor._min_sample_shading, 0.0f, 1.0f));
        gl_assert(glapi, rasterizer_state::apply() after glMinSampleShading);
    }

    if (_descriptor._scissor_test != in_applied_state._descriptor._scissor_test) {
        if (_descriptor._scissor_test) {
            glapi.glEnable(GL_SCISSOR_TEST);
        }
        else {
            glapi.glDisable(GL_SCISSOR_TEST);
        }
        gl_assert(glapi, rasterizer_state::apply() after GL_SCISSOR_TEST);
    }
    if (_descriptor._smooth_lines != in_applied_state._descriptor._smooth_lines) {
        if (_descriptor._smooth_lines) {
            glapi.glEnable(GL_LINE_SMOOTH);
        }
        else {
            glapi.glDisable(GL_LINE_SMOOTH);
        }
        gl_assert(glapi, rasterizer_state::apply() after GL_LINE_SMOOTH);
    }

    if (in_line_width != in_applied_line_width) { // i do no care about float compare at this point
        glapi.glLineWidth(math::max<float>(0.0f, in_line_width));
        gl_assert(glapi, rasterizer_state::apply() after glLineWidth);
    }
    if (in_point_size != in_applied_point_size) { // i do no care about float compare at this point
        glapi.glPointSize(math::max<float>(0.0f, in_point_size));
        gl_assert(glapi, rasterizer_state::apply() after glPointSize);
    }
    if (_descriptor._point_state != in_applied_state._descriptor._point_state) {
        if (_descriptor._point_state._shader_point_size) {
            glapi.glEnable(GL_PROGRAM_POINT_SIZE);
        }
        else {
            glapi.glDisable(GL_PROGRAM_POINT_SIZE);
        }
        glapi.glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, math::max<float>(0.0f, _descriptor._point_state._point_fade_threshold));
        glapi.glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, util::gl_origin_mode( _descriptor._point_state._point_origin_mode));
        gl_assert(glapi, rasterizer_state::apply() after glPointParameter);
    }

    gl_assert(glapi, leaving rasterizer_state::apply());
}

void
rasterizer_state::force_apply(const render_context&   in_context,
                              const float             in_line_width,
                              const float             in_point_size) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    glapi.glPolygonMode(GL_FRONT_AND_BACK, util::gl_fill_mode(_descriptor._fill_mode));
    gl_assert(glapi, rasterizer_state::force_apply() after glPolygonMode);
    
    if (_descriptor._cull_mode == CULL_NONE) {
        glapi.glDisable(GL_CULL_FACE);
        gl_assert(glapi, rasterizer_state::force_apply() after glCullFace);
    }
    else {
        glapi.glEnable(GL_CULL_FACE);
        glapi.glCullFace(util::gl_cull_mode(_descriptor._cull_mode));
    }
    glapi.glFrontFace(util::gl_polygon_orientation(_descriptor._front_face));
    gl_assert(glapi, rasterizer_state::force_apply() after glFrontFace);

    if (_descriptor._multi_sample) {
        glapi.glEnable(GL_MULTISAMPLE);
    }
    else {
        glapi.glDisable(GL_MULTISAMPLE);
    }
    gl_assert(glapi, rasterizer_state::force_apply() after GL_MULTISAMPLE);

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        if (   _descriptor._multi_sample
            && _descriptor._sample_shading)
        {
            glapi.glEnable(GL_SAMPLE_SHADING);
        }
        else {
            glapi.glDisable(GL_SAMPLE_SHADING);
        }
        gl_assert(glapi, rasterizer_state::force_apply() after GL_SAMPLE_SHADING);

        glapi.glMinSampleShading(math::clamp(_descriptor._min_sample_shading, 0.0f, 1.0f));
        gl_assert(glapi, rasterizer_state::force_apply() after glMinSampleShading);
    }

    if (_descriptor._scissor_test) {
        glapi.glEnable(GL_SCISSOR_TEST);
    }
    else {
        glapi.glDisable(GL_SCISSOR_TEST);
    }
    gl_assert(glapi, rasterizer_state::force_apply() after GL_SCISSOR_TEST);

    if (_descriptor._smooth_lines) {
        glapi.glEnable(GL_LINE_SMOOTH);
    }
    else {
        glapi.glDisable(GL_LINE_SMOOTH);
    }
    gl_assert(glapi, rasterizer_state::force_apply() after GL_LINE_SMOOTH);

    glapi.glLineWidth(math::max<float>(1.0f, in_line_width));
    gl_assert(glapi, rasterizer_state::force_apply() after glLineWidth);

    glapi.glPointSize(math::max<float>(0.0f, in_point_size));
    gl_assert(glapi, rasterizer_state::force_apply() after glPointSize);

    if (_descriptor._point_state._shader_point_size) {
        glapi.glEnable(GL_PROGRAM_POINT_SIZE);
    }
    else {
        glapi.glDisable(GL_PROGRAM_POINT_SIZE);
    }
    glapi.glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, math::max<float>(0.0f, _descriptor._point_state._point_fade_threshold));
    glapi.glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, util::gl_origin_mode( _descriptor._point_state._point_origin_mode));
    gl_assert(glapi, rasterizer_state::force_apply() after glPointParameter);

    gl_assert(glapi, leaving rasterizer_state::force_apply());
}

} // namespace gl
} // namespace scm
