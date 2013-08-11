
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_SAMPLER_STATE_H_INCLUDED
#define SCM_GL_CORE_SAMPLER_STATE_H_INCLUDED

#include <limits>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) sampler_state_desc
{
    sampler_state_desc(texture_filter_mode  in_filter = FILTER_MIN_MAG_MIP_LINEAR,
                       texture_wrap_mode    in_wrap_s = WRAP_CLAMP_TO_EDGE,
                       texture_wrap_mode    in_wrap_t = WRAP_CLAMP_TO_EDGE,
                       texture_wrap_mode    in_wrap_r = WRAP_CLAMP_TO_EDGE,
                       unsigned             in_max_anisotropy = 1,
                       float                in_min_lod = -(std::numeric_limits<float>::max)(),
                       float                in_max_lod = (std::numeric_limits<float>::max)(),
                       float                in_lod_bias = 0.0f,
                       compare_func         in_compare_func = COMPARISON_LESS_EQUAL,
                       texture_compare_mode in_compare_mode = TEXCOMPARE_NONE);

    texture_filter_mode     _filter;
    unsigned                _max_anisotropy;
    texture_wrap_mode       _wrap_s;
    texture_wrap_mode       _wrap_t;
    texture_wrap_mode       _wrap_r;
    float                   _min_lod;
    float                   _max_lod;
    float                   _lod_bias;

    compare_func            _compare_func;
    texture_compare_mode    _compare_mode;
}; // struct sampler_state_desc

class __scm_export(gl_core) sampler_state : public render_device_child
{
public:
    virtual ~sampler_state();

    const sampler_state_desc&   descriptor() const;

    unsigned                    sampler_id() const;

protected:
    sampler_state(render_device&            in_device,
                  const sampler_state_desc& in_desc);

    void                        bind(const render_context&     in_context,
                                     const int                 in_unit) const;
    void                        unbind(const render_context&   in_context,
                                       const int               in_unit) const;

protected:
    sampler_state_desc          _descriptor;
    unsigned                    _gl_sampler_id;

private:
    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
    friend class scm::gl::texture;
}; // class sampler_state

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_SAMPLER_STATE_H_INCLUDED
