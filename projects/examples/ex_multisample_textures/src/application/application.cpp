
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "application.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>
#include <scm/core/math.h>

#include <scm/gl_core.h>
#include <scm/gl_core/math.h>

#include <scm/gl_util/data/imaging/texture_loader.h>
#include <scm/gl_util/data/imaging/texture_loader_dds.h>
#include <scm/gl_util/primitives/fullscreen_triangle.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/primitives/wavefront_obj.h>
#include <scm/gl_util/utilities/coordinate_cross.h>

namespace {

const scm::math::vec3f diffuse(0.7f, 0.7f, 0.7f);
const scm::math::vec3f specular(0.2f, 0.7f, 0.9f);
const scm::math::vec3f ambient(0.1f, 0.1f, 0.1f);
const scm::math::vec3f position(1, 1, 1);

struct scm_debug_output : public scm::gl::render_context::debug_output
{
    void operator()(scm::gl::debug_source   src,
                    scm::gl::debug_type     t,
                    scm::gl::debug_severity sev,
                    const std::string&      msg) const
    {
        using namespace scm;
        using namespace scm::gl;
        out() << log::error
              << "gl error: <source: " << debug_source_string(src)
              << ", type: "            << debug_type_string(t)
              << ", severity: "        << debug_severity_string(sev) << "> "
              << msg << log::end;
    }
};

} // namespace

namespace scm {
namespace gl {
namespace gui {

application_window::application_window(const math::vec2ui&                    vp_size,
                                       const gl::viewer::viewer_attributes&   view_attrib,
                                       const gl::wm::context::attribute_desc& ctx_attrib,
                                       const gl::wm::surface::format_desc&    win_fmt)
  : viewer_window(vp_size, view_attrib, ctx_attrib, win_fmt)
  , _ms_samples(4)
{
    if (!init_renderer()) {
        std::stringstream msg;
        msg << "application_window::application_window(): error initializing multi large image rendering stystem.";
        err() << msg.str() << log::end;
        throw (std::runtime_error(msg.str()));
    }
}

application_window::~application_window()
{
    shutdown();
    std::cout << "application_window::~application_window(): bye, bye..." << std::endl;
}


bool
application_window::init_renderer()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    const render_device_ptr& device = _viewer->device();

    device->main_context()->register_debug_callback(make_shared<scm_debug_output>());


    if (!reload_shaders(device)) {
        scm::err() << log::error
                   << "application_window::init_renderer(): "
                   << "error loading program" << log::end;
        return false;
    }

    // ms texture stuff
    _ms_color = device->create_texture_2d(_viewport_size, FORMAT_RGBA_8, 1/*mip levels*/, 1/*array layers*/, _ms_samples/*samples*/);
    _ms_depth = device->create_texture_2d(_viewport_size, FORMAT_D24_S8, 1/*mip levels*/, 1/*array layers*/, _ms_samples/*samples*/);

    if (   !_ms_color
        || !_ms_depth) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error creating multi sample texture." << log::end;
        return false;
    }

    _ms_fbo = device->create_frame_buffer();

    if (!_ms_fbo) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error creating multi sample frame buffer." << log::end;
        return false;
    }

    _ms_fbo->attach_color_buffer(0, _ms_color);
    _ms_fbo->attach_depth_stencil_buffer(_ms_depth);

    _resolve_bstate = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _resolve_dstate = device->create_depth_stencil_state(false, false, COMPARISON_LESS);
    _resolve_rstate = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, false);
    _resolve_sstate = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);

    try {
        _resolve_quad = make_shared<quad_geometry>(device, vec2f::zero(), vec2f::one());
        _resolve_tri  = make_shared<fullscreen_triangle>(device);
    }
    catch(std::exception& e) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error generating resolve quad geometry: " << e.what() << log::end;
        return false;
    }

    // model stuff
    _model_bstate = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _model_dstate = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _model_rstate = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);
    _model_sstate = device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);

    try {
        texture_loader tl;
        _env_map           = tl.load_texture_2d(*device, "../../../res/textures/mobile_ave_env.hdr", false);
        _env_map_diff      = tl.load_texture_2d(*device, "../../../res/textures/mobile_ave_diff.hdr", false);
        _model_normal_map  = tl.load_texture_2d(*device, "../../../res/textures/guardian_normals.tif", false);
        _model_diffuse_map = tl.load_texture_2d(*device, "../../../res/textures/guardian_diffuse.tif", false);
        _model             = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/guardian.obj");
    }
    catch(std::exception& e) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error loading model geometry: " << e.what() << log::end;
        return false;
    }

    if (   !_env_map
        || !_env_map_diff
        || !_model_normal_map
        || !_model_diffuse_map) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error loading textures." << log::end;
        return false;
    }

    _animation_enabled = false;
    _rotation_angle    = 0.0f;

    try {
        _coord_cross = make_shared<coordinate_cross>(device, 0.15f);
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "application_window::init_renderer(): unable to initialize the render system ("
            << "evoking error: " << e.what() << ").";
        err() << msg.str() << log::end;
        return (false);
    }

    return (true);
}

bool
application_window::reload_shaders(const gl::render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    { // model program
        program_ptr p  = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/shaders/simple.glslv"))
                                                       (device->create_shader_from_file(STAGE_GEOMETRY_SHADER, "../../../src/shaders/simple.glslg"))
                                                       (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/shaders/simple.glslf")),
                                                "model_prog");

        if (!p) {
            scm::err() << log::error
                       << "application_window::reload_shaders(): "
                       << "error loading program" << log::end;
            return false;
        }
        else {
            _model_prog = p;
        }
    }
    { // resolve program
        program_ptr p  = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/shaders/resolve.glslv"))
                                                       (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/shaders/resolve.glslf")),
                                                "resolve_prog");

        if (!p) {
            scm::err() << log::error
                       << "application_window::reload_shaders(): "
                       << "error loading program" << log::end;
            return false;
        }
        else {
            mat4f pass_mvp = make_ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
            p->uniform("mvp", pass_mvp);
            p->uniform("num_samples", _ms_samples);
            _resolve_prog = p;
        }
    }

    return true;
}

void
application_window::shutdown()
{
    _coord_cross.reset();
}

void
application_window::update(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context)
{
    float deg_per_sec = 180.0f;
    if (_animation_enabled) {
        _rotation_angle += deg_per_sec * (_viewer->frame_time_us() / 1e6f);
    }
}

void
application_window::display(const gl::render_context_ptr& context)
{
    using namespace scm::gl;
    using namespace scm::math;

    const mat4f  rotation_matrix     = make_rotation(_rotation_angle, vec3f(0.0f, 1.0f, 0.0f));
    const mat4f  model_matrix        = make_scale(vec3f(0.2f)) * rotation_matrix;
    const mat4f& view_matrix         = _viewer->main_camera().view_matrix() * model_matrix;
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    { // ms pass
        context_program_guard       cpg(context);
        context_state_objects_guard csg(context);
        context_texture_units_guard tsg(context);
        context_framebuffer_guard   fsg(context);

        context->clear_color_buffer(_ms_fbo, 0, vec4f(0.2f));
        context->clear_depth_stencil_buffer(_ms_fbo);

        _model_prog->uniform("projection_matrix", proj_matrix);
        _model_prog->uniform("model_view_matrix", view_matrix);

        context->set_frame_buffer(_ms_fbo);
        context->set_viewport(viewport(vec2ui::zero(), _viewport_size));

        context->bind_texture(_env_map,           _model_sstate, 0/*texture unit 0*/);
        context->bind_texture(_env_map_diff,      _model_sstate, 1/*texture unit 1*/);
        context->bind_texture(_model_normal_map,  _model_sstate, 2/*texture unit 2*/);
        context->bind_texture(_model_diffuse_map, _model_sstate, 3/*texture unit 3*/);

        context->set_rasterizer_state(_model_rstate);
        context->set_blend_state(_model_bstate);
        context->set_depth_stencil_state(_model_dstate);

        context->bind_program(_model_prog);

        _model->draw_raw(context);
        _coord_cross->draw(context, proj_matrix, view_matrix, 4.0f);
    }

    { // resolve pass
        context_state_objects_guard csg(context);
        context_texture_units_guard tug(context);
        context_framebuffer_guard   fbg(context);

        context->set_depth_stencil_state(_resolve_dstate);
        context->set_blend_state(_resolve_bstate);
        context->set_rasterizer_state(_resolve_rstate);

        context->bind_program(_resolve_prog);

        context->bind_texture(_ms_color, _resolve_sstate, 0/*texture unit 0*/);

        _resolve_tri->draw(context, geometry::MODE_SOLID);
        //_resolve_quad->draw(context, geometry::MODE_SOLID);
    }

    if (0) {
        context_program_guard       cpg(context);
        context_state_objects_guard csg(context);
        context_texture_units_guard tsg(context);

        _model_prog->uniform("projection_matrix", proj_matrix);
        _model_prog->uniform("model_view_matrix", view_matrix);

        context->bind_program(_model_prog);

        context->bind_texture(_env_map,           _model_sstate, 0);
        context->bind_texture(_env_map_diff,      _model_sstate, 1);
        context->bind_texture(_model_normal_map,  _model_sstate, 2);
        context->bind_texture(_model_diffuse_map, _model_sstate, 3);

        context->set_rasterizer_state(_model_rstate);
        context->set_blend_state(_model_bstate);
        context->set_depth_stencil_state(_model_dstate);

        _model->draw_raw(context);
    }
}

void
application_window::reshape(const gl::render_device_ptr& device,
                            const gl::render_context_ptr& context,
                            int w, int h)
{
    viewer_window::reshape(device, context, w, h);
}

void
application_window::keyboard_input(int k, bool state, scm::uint32 mod)
{
    viewer_window::keyboard_input(k, state, mod);
    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_Escape:    close_program();break;
            case Qt::Key_Space:     _animation_enabled = !_animation_enabled;break;
            case Qt::Key_S:         reload_shaders(_viewer->device());break;
            default:;
        }
    }
}

void
application_window::mouse_double_click(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_double_click()" << log::end;
}

void
application_window::mouse_press(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_press()" << log::end;
}

void
application_window::mouse_release(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_release()" << log::end;
}

void
application_window::mouse_move(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_move()" << log::end;
}

} // namespace gui
} // namespace gl
} // namespace scm
