
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "height_field_tessellator.h"

#include <exception>
#include <stdexcept>
#include <string>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/viewer/camera_uniform_block.h>

#include <renderers/height_field_data.h>
#include <renderers/patch_grid_mesh.h>

namespace scm {
namespace data {

height_field_tessellator::height_field_tessellator(const gl::render_device_ptr& device)
  : _pixel_tolerance(4.0f)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    // state objects //////////////////////////////////////////////////////////////////////////////
    _bstate_no_blend     = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _bstate_omsa         = device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);
    _dstate_less         = device->create_depth_stencil_state(true, true, COMPARISON_LESS);

    _rstate_ms_wire      = device->create_rasterizer_state(FILL_WIREFRAME, CULL_NONE, ORIENT_CCW, true);
    _rstate_ms_solid     = device->create_rasterizer_state(FILL_SOLID, CULL_NONE, ORIENT_CCW, true);
    _rstate_ms_solid_ss  = device->create_rasterizer_state(FILL_SOLID, CULL_NONE, ORIENT_CCW, true, true, 1.0f);
    //_rstate_ms_wire      = device->create_rasterizer_state(FILL_WIREFRAME, CULL_BACK, ORIENT_CCW, true);
    //_rstate_ms_solid     = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    _sstate_nearest      = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _sstate_linear       = device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);
    _sstate_linear_mip   = device->create_sampler_state(FILTER_MIN_MAG_MIP_LINEAR, WRAP_CLAMP_TO_EDGE);

    if (   !_bstate_no_blend
        || !_bstate_omsa
        || !_dstate_less
        || !_rstate_ms_wire
        || !_rstate_ms_solid
        || !_sstate_nearest
        || !_sstate_linear
        || !_sstate_linear_mip) {
        throw (std::runtime_error("height_field_tessellator::height_field_tessellator(): error creating state objects"));
    }

    gl::camera_uniform_block::add_block_include_string(device);
    device->add_include_files("./../../../src/renderers/shaders", "/scm/data");

    // ray casting program ////////////////////////////////////////////////////////////////////////
    _hf_quad_tessellation_program = device->create_program(
        list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,          "../../../src/renderers/shaders/height_field/tessellate.glslv"))
               (device->create_shader_from_file(STAGE_TESS_CONTROL_SHADER,    "../../../src/renderers/shaders/height_field/tessellate_quad.glsltc"))
               (device->create_shader_from_file(STAGE_TESS_EVALUATION_SHADER, "../../../src/renderers/shaders/height_field/tessellate_quad.glslte"))
               (device->create_shader_from_file(STAGE_GEOMETRY_SHADER,        "../../../src/renderers/shaders/height_field/tessellate.glslg"))
               (device->create_shader_from_file(STAGE_FRAGMENT_SHADER,        "../../../src/renderers/shaders/height_field/tessellate.glslf")),
        "height_field_tessellator::hf_quad_tessellation_program");
    _hf_triangle_tessellation_program = device->create_program(
        list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,          "../../../src/renderers/shaders/height_field/tessellate.glslv"))
               (device->create_shader_from_file(STAGE_TESS_CONTROL_SHADER,    "../../../src/renderers/shaders/height_field/tessellate_tri.glsltc", shader_macro("PRIMITIVE_ID_REPRO", "0")))
               (device->create_shader_from_file(STAGE_TESS_EVALUATION_SHADER, "../../../src/renderers/shaders/height_field/tessellate_tri.glslte"))
               (device->create_shader_from_file(STAGE_GEOMETRY_SHADER,        "../../../src/renderers/shaders/height_field/tessellate.glslg"))
               (device->create_shader_from_file(STAGE_FRAGMENT_SHADER,        "../../../src/renderers/shaders/height_field/tessellate.glslf")),
        //separate_stream_capture("per_vertex.vertex.ws_position"),
        //interleaved_stream_capture("per_vertex.vertex.ws_position")("per_vertex.vertex.texcoord_hm"),
        //interleaved_stream_capture("per_vertex.vertex.ws_position")("per_vertex.vertex.texcoord_hm")(stream_capture::skip_4_float), // GL4+ only
        //stream_capture_array("per_vertex.vertex.ws_position")
        //stream_capture_array("per_vertex.vertex.ws_position")("per_vertex.vertex.texcoord_hm")("per_vertex.vertex.texcoord_dm"),
        //stream_capture_array(interleaved_stream_capture("per_vertex.vertex.ws_position")("per_vertex.vertex.texcoord_hm"))
        //                    (separate_stream_capture("per_vertex.vertex.texcoord_dm")),
        //stream_capture_array(interleaved_stream_capture("per_vertex.vertex.ws_position")("per_vertex.vertex.texcoord_hm")(stream_capture::skip_4_float))
        //                    (separate_stream_capture("per_vertex.vertex.texcoord_dm")),
        "height_field_tessellator::hf_triangle_tessellation_program");

    if (   !_hf_quad_tessellation_program
        || !_hf_triangle_tessellation_program) {
        throw (std::runtime_error("height_field_ray_caster::height_field_ray_caster(): error creating quad or triangle program"));
    }

    // set default uniforms
    _hf_quad_tessellation_program->uniform_buffer("camera_matrices", 0);
    _hf_quad_tessellation_program->uniform_sampler("height_map",          0);
    _hf_quad_tessellation_program->uniform_sampler("height_map_nearest",  1);
    _hf_quad_tessellation_program->uniform_sampler("density_map",         2);
    _hf_quad_tessellation_program->uniform_sampler("color_map",           3);

    // set default uniforms
    _hf_triangle_tessellation_program->uniform_buffer("camera_matrices", 0);
    _hf_triangle_tessellation_program->uniform_sampler("height_map",         0);
    _hf_triangle_tessellation_program->uniform_sampler("height_map_nearest", 1);
    _hf_triangle_tessellation_program->uniform_sampler("density_map",        2);
    _hf_triangle_tessellation_program->uniform_sampler("color_map",          3);
    _hf_triangle_tessellation_program->uniform_sampler("edge_densities",     4);

    _main_camera_block.reset(new gl::camera_uniform_block(device));
}

height_field_tessellator::~height_field_tessellator()
{
    _hf_quad_tessellation_program.reset();
    _hf_triangle_tessellation_program.reset();

    _bstate_no_blend.reset();
    _bstate_omsa.reset();
    _dstate_less.reset();
    _rstate_ms_wire.reset();
    _rstate_ms_solid.reset();
    _sstate_nearest.reset();
    _sstate_linear.reset();
    _sstate_linear_mip.reset();

    _main_camera_block.reset();
}

float
height_field_tessellator::pixel_tolerance() const
{
    return _pixel_tolerance;
}

void
height_field_tessellator::pixel_tolerance(float t)
{
    _pixel_tolerance = math::max(0.25f, t);
}

void
height_field_tessellator::update_main_camera(const gl::render_context_ptr& context,
                                             const gl::camera&             cam)
{
    using namespace scm::gl;
    using namespace scm::math;

    _main_camera_block->update(context, cam);
}

void
height_field_tessellator::draw(const gl::render_context_ptr& context,
                               const height_field_data_ptr&  hf_data,
                                     bool                    super_sample,
                               const mesh_mode               hf_mesh_mode,
                               const draw_mode               hf_draw_mode) const
{
    using namespace scm::gl;
    using namespace scm::math;

    static float tessellation_factor = 1.2f;
    static float tessellation_inc    = 0.01f;

    context_state_objects_guard     csg(context);
    context_program_guard           cpg(context);
    context_uniform_buffer_guard    ubg(context);
    context_texture_units_guard     tug(context);

    tessellation_factor += tessellation_inc;

    if (tessellation_factor >= 32.0) tessellation_inc = -tessellation_inc;
    if (tessellation_factor <  1.1)  tessellation_inc = -tessellation_inc;


    context->bind_uniform_buffer(_main_camera_block->block().block_buffer(), 0);

    context->bind_texture(hf_data->height_map(),  _sstate_linear_mip, 0);
    context->bind_texture(hf_data->height_map(),  _sstate_nearest,    1);
    context->bind_texture(hf_data->density_map(), _sstate_nearest,    2);
    context->bind_texture(hf_data->color_map(),   _sstate_linear,     3);

    context->set_depth_stencil_state(_dstate_less);
    context->set_blend_state(_bstate_no_blend);
    //context->set_blend_state(_bstate_omsa);

    if (hf_draw_mode == MODE_SOLID) {
        context->set_rasterizer_state(super_sample ? _rstate_ms_solid_ss : _rstate_ms_solid);
    }
    else if (hf_draw_mode == MODE_WIRE_FRAME) {
        context->set_rasterizer_state(_rstate_ms_wire);
    }

    if (hf_mesh_mode == MODE_QUAD_PATCHES) {
        _hf_quad_tessellation_program->uniform("tessellation_factor",  5.0f);//tessellation_factor);
        _hf_quad_tessellation_program->uniform("model_matrix",  hf_data->transform());
        _hf_quad_tessellation_program->uniform("height_scale",  hf_data->extends().z);
        _hf_quad_tessellation_program->uniform("pixel_tolerance",  _pixel_tolerance);
        _hf_quad_tessellation_program->uniform("screen_size",  vec2f(1920.0f, 1200.0f));

        context->bind_program(_hf_quad_tessellation_program);
            
        hf_data->patch_mesh()->draw(context, patch_grid_mesh::MODE_QUAD_PATCHES);
    }
    else if (hf_mesh_mode == MODE_TRIANGLE_PATCHES) {
        _hf_triangle_tessellation_program->uniform("tessellation_factor",  5.0f);//tessellation_factor);
        _hf_triangle_tessellation_program->uniform("model_matrix",  hf_data->transform());
        _hf_triangle_tessellation_program->uniform("height_scale",  hf_data->extends().z);
        _hf_triangle_tessellation_program->uniform("pixel_tolerance",  _pixel_tolerance);
        _hf_triangle_tessellation_program->uniform("screen_size",  vec2f(1920.0f, 1200.0f));

        context->bind_texture(hf_data->triangle_edge_density_buffer(), _sstate_nearest, 4);
        context->bind_program(_hf_triangle_tessellation_program);
            
        hf_data->patch_mesh()->draw(context, patch_grid_mesh::MODE_TRIANGLE_PATCHES);
    }
    else {
        err() << log::warning << "height_field_tessellator::draw(): unknown mesh mode." << log::end;
    }
}

} // namespace data
} // namespace scm
