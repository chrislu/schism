
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LDATA_HEIGHT_FIELD_TESSELLATOR_H_INCLUDED
#define SCM_LDATA_HEIGHT_FIELD_TESSELLATOR_H_INCLUDED

#include <scm/core/math.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/constants.h>
#include <scm/gl_core/buffer_objects/uniform_buffer_adaptor.h>

#include <scm/gl_util/viewer/viewer_fwd.h>

#include <renderers/renderers_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace data {

class /*__scm_export(large_data*/ height_field_tessellator
{
public:
    enum draw_mode {
        MODE_SOLID      = 0x00,
        MODE_WIRE_FRAME,
    };
    enum mesh_mode {
        MODE_QUAD_PATCHES       = 0x00,
        MODE_TRIANGLE_PATCHES
    };


public:
    height_field_tessellator(const gl::render_device_ptr& device);
    virtual ~height_field_tessellator();

    float                           pixel_tolerance() const;
    void                            pixel_tolerance(float t);

    void                            update_main_camera(const gl::render_context_ptr& context,
                                                       const gl::camera&             cam);
    void                            draw(const gl::render_context_ptr& context,
                                         const height_field_data_ptr&  hf_data,
                                               bool                    super_sample = false,
                                         const mesh_mode               hf_mesh_mode = MODE_QUAD_PATCHES,
                                         const draw_mode               hf_draw_mode = MODE_SOLID) const;

protected:
    // programs
    gl::program_ptr                 _hf_quad_tessellation_program;
    gl::program_ptr                 _hf_triangle_tessellation_program;

    // state objects
    gl::blend_state_ptr             _bstate_no_blend;
    gl::blend_state_ptr             _bstate_omsa;
    gl::depth_stencil_state_ptr     _dstate_less;
    gl::rasterizer_state_ptr        _rstate_ms_wire;
    gl::rasterizer_state_ptr        _rstate_ms_solid;
    gl::rasterizer_state_ptr        _rstate_ms_solid_ss;
    gl::sampler_state_ptr           _sstate_nearest;
    gl::sampler_state_ptr           _sstate_linear;
    gl::sampler_state_ptr           _sstate_linear_mip;

    // transform feedback
    gl::transform_feedback_ptr      _tess_feedback;

    // uniform blocks
    gl::camera_uniform_block_ptr    _main_camera_block;

    // uniform values
    float                           _pixel_tolerance; // clamp (max(0.5, x)

}; // height_field_tessellator

} // namespace data
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_LDATA_HEIGHT_FIELD_TESSELLATOR_H_INCLUDED
