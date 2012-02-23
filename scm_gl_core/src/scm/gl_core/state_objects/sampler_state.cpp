
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "sampler_state.h"

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

sampler_state_desc::sampler_state_desc(
    texture_filter_mode      in_filter,
    texture_wrap_mode        in_wrap_s,
    texture_wrap_mode        in_wrap_t,
    texture_wrap_mode        in_wrap_r,
    unsigned                 in_max_anisotropy,
    float                    in_min_lod,
    float                    in_max_lod,
    float                    in_lod_bias,
    compare_func             in_compare_func,
    texture_compare_mode     in_compare_mode)
  : _filter(in_filter)
  , _max_anisotropy(in_max_anisotropy)
  , _wrap_s(in_wrap_s)
  , _wrap_t(in_wrap_t)
  , _wrap_r(in_wrap_r)
  , _min_lod(in_min_lod)
  , _max_lod(in_max_lod)
  , _lod_bias(in_lod_bias)
  , _compare_func(in_compare_func)
  , _compare_mode(in_compare_mode)
{
}

// sampler_state //////////////////////////////////////////////////////////////////////////////////
sampler_state::sampler_state(render_device&            in_device,
                             const sampler_state_desc& in_desc)
  : render_device_child(in_device)
  , _descriptor(in_desc)
  , _gl_sampler_id(0)
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    glapi.glGenSamplers(1, &_gl_sampler_id);
    if (0 == _gl_sampler_id) {
        state().set(object_state::OS_BAD);
    }
    else {

        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_MAG_FILTER,
                                  util::gl_texture_mag_filter_mode(descriptor()._filter));
        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_MIN_FILTER,
                                  util::gl_texture_min_filter_mode(descriptor()._filter));
        if (descriptor()._filter == FILTER_ANISOTROPIC) {
            glapi.glSamplerParameterf(sampler_id(), GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                      static_cast<float>(descriptor()._max_anisotropy));
        }
        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_WRAP_S, util::gl_wrap_mode(descriptor()._wrap_s));
        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_WRAP_T, util::gl_wrap_mode(descriptor()._wrap_t));
        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_WRAP_R, util::gl_wrap_mode(descriptor()._wrap_r));
        glapi.glSamplerParameterf(sampler_id(), GL_TEXTURE_MIN_LOD, descriptor()._min_lod);
        glapi.glSamplerParameterf(sampler_id(), GL_TEXTURE_MAX_LOD, descriptor()._max_lod);
        glapi.glSamplerParameterf(sampler_id(), GL_TEXTURE_LOD_BIAS, descriptor()._lod_bias);
        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_COMPARE_FUNC, util::gl_compare_func(descriptor()._compare_func));
        glapi.glSamplerParameteri(sampler_id(), GL_TEXTURE_COMPARE_MODE, util::gl_texture_compare_mode(descriptor()._compare_mode));
    }

    gl_assert(glapi, leaving sampler_state::sampler_state());
}

sampler_state::~sampler_state()
{
    const opengl::gl_core& glapi = parent_device().opengl_api();

    assert(0 != _gl_sampler_id);
    glapi.glDeleteSamplers(1, &_gl_sampler_id);
    
    gl_assert(glapi, leaving sampler_state::~sampler_state());
}

const sampler_state_desc&
sampler_state::descriptor() const
{
    return (_descriptor);
}

unsigned
sampler_state::sampler_id() const
{
    return (_gl_sampler_id);
}

void
sampler_state::bind(const render_context&     in_context,
                    const int                 in_unit) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != sampler_id());

    glapi.glBindSampler(in_unit, sampler_id());

    gl_assert(glapi, leaving sampler_state::bind());
}

void
sampler_state::unbind(const render_context& in_context,
                      const int             in_unit) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    glapi.glBindSampler(in_unit, 0);

    gl_assert(glapi, leaving sampler_state::unbind());
}

} // namespace gl
} // namespace scm
