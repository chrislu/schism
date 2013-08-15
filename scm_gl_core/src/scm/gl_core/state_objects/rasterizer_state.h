
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_RASTERIZER_STATE_H_INCLUDED
#define SCM_GL_CORE_RASTERIZER_STATE_H_INCLUDED

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) point_raster_state {
    point_raster_state(bool            in_shader_point_size    = false,
                       origin_mode     in_point_origin         = ORIGIN_LOWER_LEFT,
                       float           in_point_fade_threshold = 1.0f);

    bool operator==(const point_raster_state& rhs) const;
    bool operator!=(const point_raster_state& rhs) const;

    bool            _shader_point_size;
    origin_mode     _point_origin_mode;
    float           _point_fade_threshold;

}; // struct point_raster_state

struct __scm_export(gl_core) rasterizer_state_desc {
    rasterizer_state_desc(fill_mode                 in_fmode = FILL_SOLID,
                          cull_mode                 in_cmode = CULL_BACK,
                          polygon_orientation       in_fface = ORIENT_CCW,
                          bool                      in_msample = false,
                          bool                      in_sshading = false,
                          float32                   in_min_sshading = 0.0f,
                          bool                      in_sctest = false,
                          bool                      in_smlines = false,
                          const point_raster_state& in_point_state = point_raster_state());

    fill_mode               _fill_mode;
    cull_mode               _cull_mode;

    polygon_orientation     _front_face;

    bool                    _multi_sample;
    bool                    _sample_shading;
    float32                 _min_sample_shading;

    bool                    _scissor_test;
    bool                    _smooth_lines;

    point_raster_state      _point_state;
}; // struct depth_stencil_state_desc

class __scm_export(gl_core) rasterizer_state : public render_device_child
{
public:
    virtual ~rasterizer_state();

    const rasterizer_state_desc&    descriptor() const;

protected:
    rasterizer_state(      render_device&         in_device,
                     const rasterizer_state_desc& in_desc);

    void                            apply(const render_context&   in_context,
                                          const float             in_line_width,
                                          const float             in_point_size,
                                          const rasterizer_state& in_applied_state,
                                          const float             in_applied_line_width,
                                          const float             in_applied_point_size) const;
    void                            force_apply(const render_context&   in_context,
                                                const float             in_line_width,
                                                const float             in_point_size) const;

protected:
    rasterizer_state_desc           _descriptor;

private:
    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
}; // class rasterizer_state

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_RASTERIZER_STATE_H_INCLUDED
