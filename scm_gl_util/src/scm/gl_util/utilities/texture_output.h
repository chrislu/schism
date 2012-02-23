
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_MVG_DEBUG_VISUALIZERS_H_INCLUDED
#define SCM_MVG_DEBUG_VISUALIZERS_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) texture_output
{
public:
    texture_output(const gl::render_device_ptr& device);
    virtual ~texture_output();

    void draw_texture_2d(const gl::render_context_ptr& context,
                         const gl::texture_2d_ptr&     tex,
                         const math::vec2ui&           position,
                         const math::vec2ui&           extend) const;
    void draw_texture_2d_uint(const gl::render_context_ptr& context,
                              const gl::texture_2d_ptr&     tex,
                              const math::vec4f&            scale,
                              const math::vec2ui&           position,
                              const math::vec2ui&           extend) const;
    void draw_texture_2d_uint8_bit_rev(const gl::render_context_ptr& context,
                                       const gl::texture_2d_ptr&     tex,
                                       const math::vec2ui&           position,
                                       const math::vec2ui&           extend) const;
 
protected:
    gl::quad_geometry_ptr           _quad_geom;
    gl::program_ptr                 _fs_program_color;
    gl::program_ptr                 _fs_program_color_uint;
    gl::program_ptr                 _fs_program_color_uint8_bit_rev;
    gl::program_ptr                 _fs_program_gray;
    gl::program_ptr                 _fs_program_gray_uint;
    gl::program_ptr                 _fs_program_gray_uint8_bit_rev;

    gl::sampler_state_ptr           _filter_nearest;
    gl::rasterizer_state_ptr        _rstate_cull_back;
    gl::blend_state_ptr             _bstate_no_blend;
    gl::depth_stencil_state_ptr     _dstate_no_z_write;

}; // class texture_output

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_MVG_DEBUG_VISUALIZERS_H_INCLUDED
