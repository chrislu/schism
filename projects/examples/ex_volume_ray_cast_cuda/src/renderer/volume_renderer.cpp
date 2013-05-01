
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_renderer.h"

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>

#include <boost/assign/list_of.hpp>

#include <scm/gl_core/math.h>
#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/box_volume.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/viewer/camera_uniform_block.h>

#include <renderer/volume_data.h>

namespace scm {
namespace data {

volume_renderer::volume_renderer(const gl::render_device_ptr& device, const math::vec2ui& vp_size)
  : _viewport_size(vp_size)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    camera_uniform_block::add_block_include_string(device);

    _camera_block.reset(new gl::camera_uniform_block(device));

    // state objects //////////////////////////////////////////////////////////////////////////////
    _bstate = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    //_bstate = device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);
    _dstate = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _rstate = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);
    _sstate_lin = device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);
    _sstate_lin_mip = device->create_sampler_state(FILTER_MIN_MAG_MIP_LINEAR, WRAP_CLAMP_TO_EDGE);

    if (   !_bstate
        || !_dstate
        || !_rstate
        || !_sstate_lin
        || !_sstate_lin_mip) {
        throw (std::runtime_error("volume_renderer::volume_renderer(): error creating state objects"));
    }

    if (!reload_shaders(device)) {
        throw std::runtime_error("volume_renderer::volume_renderer(): error loading shader programs.");
    }

    //_vtexture_program->uniform("model_matrix", model_matrix);
    //_vtexture_program->uniform_buffer("camera_matrices", 0);
    //_vtexture_program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "out_color_gen", "draw_vtexture");
    //_vtexture_program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "out_color_gen", "draw_debug_atlas");
    //_vtexture_program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "out_color_gen", "draw_atlas");
    //_vtexture_program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "out_color_gen", "visualize_normal");
    //_vtexture_program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "out_color_gen", "visualize_texcoord");

}

volume_renderer::~volume_renderer()
{
    _program.reset();
    
    _dstate.reset();
    _bstate.reset();
    _rstate.reset();
    _sstate_lin.reset();
    _sstate_lin_mip.reset();

    _camera_block.reset();
}

void
volume_renderer::draw(const gl::render_context_ptr& context,
                      const volume_data_ptr&        vdata,
                      const vr_mode                 mode,
                      bool                          use_supersampling) const
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    switch (mode) {
        case volume_raw:
            _program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "volume_color_lookup", "raw_lookup");
            break;
        case volume_color_map:
            _program->uniform_subroutine(STAGE_FRAGMENT_SHADER, "volume_color_lookup", "raw_color_map_lookup");
            break;
    }

    _program->uniform("volume_lod", vdata->selected_lod());
    _program->uniform("use_ss",     use_supersampling);

    context_state_objects_guard     csg(context);
    context_texture_units_guard     tug(context);
    context_uniform_buffer_guard    ubg(context);

    context->reset_state_objects();

    context->set_depth_stencil_state(_dstate);
    context->set_blend_state(_bstate);
    context->set_rasterizer_state(_rstate);

    context->bind_uniform_buffer(_camera_block->block().block_buffer(), 0);
    context->bind_uniform_buffer(vdata->volume_block().block_buffer(),  1);

    context->bind_program(_program);

    context->reset_texture_units();
    context->bind_texture(vdata->volume_raw(),      _sstate_lin_mip, 0);
    context->bind_texture(vdata->color_alpha_map(), _sstate_lin,     2);

    vdata->bbox_geometry()->draw(context, geometry::MODE_SOLID);
}

void
volume_renderer::update(const gl::render_context_ptr& context,
                        const gl::camera&             cam)
{
    _camera_block->update(context, cam);
}

bool
volume_renderer::reload_shaders(const gl::render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    scm::out() << log::info
               << "volume_renderer::reload_shaders(): "
               << "reloading shader strings." << log::end;

    // ray casting program ////////////////////////////////////////////////////////////////////////
    program_ptr prog_rcraw  = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/renderer/shader/volume_ray_cast.glslv"))
                                                            (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/renderer/shader/volume_ray_cast.glslf")),
                                                     "volume_renderer::program");

    if (!prog_rcraw) {
        scm::err() << log::error
                   << "volume_renderer::reload_shaders(): "
                   << "error loading program" << log::end;
        return false;
    }
    else {
        _program = prog_rcraw;
    }

    _program->uniform("volume_raw",     0);
    _program->uniform("color_map",      2);
    _program->uniform("viewport_size",  math::vec2f(_viewport_size));

    _program->uniform_buffer("camera_matrices",     0);
    _program->uniform_buffer("volume_uniform_data", 1);

    return true;
}

} // namespace data
} // namespace scm
