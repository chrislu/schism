
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
#include <scm/gl_util/primitives/wavefront_obj.h>

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
    using boost::assign::list_of;

    const render_device_ptr& device = _viewer->device();

    device->main_context()->register_debug_callback(make_shared<scm_debug_output>());

    if (!reload_shaders(device)) {
        scm::err() << log::error
                   << "application_window::init_renderer(): "
                   << "error loading program" << log::end;
        return false;
    }

    _bstate = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _rstate = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);
    _sstate = device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);

    try {
        texture_loader tl;
        _env_map      = tl.load_texture_2d(*device, "e:/data/textures/ibl_hdr/mobile_ave_env.hdr", false);
        _env_map_diff = tl.load_texture_2d(*device, "e:/data/textures/ibl_hdr/mobile_ave_diff.hdr", false);
        //_model      = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/sphere.obj");
        //_model      = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/box.obj");
        _model      = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/guardian.obj");
        //_model      = make_shared<wavefront_obj_geometry>(device, "e:/data/geometry/beetle_felge.obj");
        //_model      = make_shared<wavefront_obj_geometry>(device, "g:/geometry/01_renault_clio.obj");
        //_model        = make_shared<wavefront_obj_geometry>(device, "e:/data/geometry/high_res_sphere3.obj");
        //_model        = make_shared<wavefront_obj_geometry>(device, "e:/data/geometry/happy_budda.obj");

        texture_loader_dds tldds;
        _model_normal_map  = tl.load_texture_2d(*device, "../../../res/geometry/guardian_normals.tif", false);
        _model_diffuse_map = tl.load_texture_2d(*device, "../../../res/geometry/guardian_diffuse.tif", false);
    }
    catch(std::exception& e) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error loading model geometry: " << e.what() << log::end;
        return false;
    }

    if (   !_env_map
        || !_env_map
        || !_model_normal_map) {
        err() << log::error
              << "application_window::init_renderer(): "
              << "error loading textures." << log::end;
        return false;
    }

    _time_query    = device->create_timer_query();
    _time_query_00 = device->create_timer_query();
    _time_query_01 = device->create_timer_query();

    _frame_number      = 0;
    _timer_start_frame = 0;

    _timer_available   = true;
    _fence_available   = true;

    _animation_enabled = false;
    _rotation_angle    = 0.0f;

    _model_grid_res     = vec3ui(3);
    _model_grid_queries = false;
    _model_grid_queries_available = true;
    _model_grid_queries_start_frame = 0;
    for (unsigned gz = 0; gz < _model_grid_res.z; ++gz) {
        for (unsigned gy = 0; gy < _model_grid_res.y; ++gy) {
            for (unsigned gx = 0; gx < _model_grid_res.x; ++gx) {
                _time_queries.push_back(device->create_timer_query());
            }
        }
    }

    _enable_output = false;

    return true;
}

bool
application_window::reload_shaders(const gl::render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    program_ptr p  = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/shaders/simple.glslv"))
                                                   (device->create_shader_from_file(STAGE_GEOMETRY_SHADER, "../../../src/shaders/simple.glslg"))
                                                   (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/shaders/simple.glslf")),
                                            "shader_prog");

    if (!p) {
        scm::err() << log::error
                   << "application_window::reload_shaders(): "
                   << "error loading program" << log::end;
        return false;
    }
    else {
        _shader_prog = p;
        return true;
    }
}

void
application_window::shutdown()
{
    _shader_prog.reset();

    _rstate.reset();
    _bstate.reset();
    _dstate.reset();
    _sstate.reset();

    _model.reset();
    _model_normal_map.reset();
    _model_diffuse_map.reset();
    _env_map.reset();
    _env_map_diff.reset();

    _time_query.reset();
    _time_query_00.reset();
    _time_query_01.reset();

    _fence.reset();
    _fence_00.reset();
    _fence_01.reset();
}

void
application_window::update(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context)
{
}

void
application_window::display(const gl::render_context_ptr& context)
{
    ++_frame_number;

    using namespace scm::gl;
    using namespace scm::math;

    float deg_per_sec = 180.0f;

    if (_animation_enabled) {
        _rotation_angle += deg_per_sec * (_viewer->frame_time_us() / 1e6f);
    }

    const mat4f rotation_matrix  = make_rotation(_rotation_angle, vec3f(0.0f, 1.0f, 0.0f));

#if 1 // check overhead of time queries
    const vec3i grid_res = vec3i(_model_grid_res);

    std::vector<mat4f>  view_matrices;

    const int   max_grid = max(grid_res.x, max(grid_res.y, grid_res.z));
    const float scale    = 0.4f / max_grid;
    const vec3f spacing  = vec3f(1.0f) / vec3f(grid_res);
    const vec3f offset   = vec3f(grid_res - 1) / 2.0f;
    const mat4f scale_mat = make_scale(vec3f(scale));
    for (int gz = 0; gz < grid_res.z; ++gz) {
        for (int gy = 0; gy < grid_res.y; ++gy) {
            for (int gx = 0; gx < grid_res.x; ++gx) {
                const vec3f o = vec3f(gx, gy, gz);
                const mat4f t = make_translation((o - offset) * spacing);
                const mat4f v = _viewer->main_camera().view_matrix() * t * rotation_matrix * scale_mat;
                view_matrices.push_back(v);
            }
        }
    }

    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    gl::uniform_mat4f_ptr view_mat_uniform = _shader_prog->uniform_mat4f("model_view_matrix");
    {
        context_program_guard       cpg(context);
        context_state_objects_guard csg(context);
        context_texture_units_guard tsg(context);

        _shader_prog->uniform("projection_matrix", proj_matrix);

        context->bind_program(_shader_prog);

        context->bind_texture(_env_map,           _sstate, 0);
        context->bind_texture(_env_map_diff,      _sstate, 1);
        context->bind_texture(_model_normal_map,  _sstate, 2);
        context->bind_texture(_model_diffuse_map, _sstate, 3);

        context->set_rasterizer_state(_rstate);
        context->set_blend_state(_bstate);
        context->set_depth_stencil_state(_dstate);

        for (size_t vs = 0; vs < view_matrices.size(); ++vs) {
            const mat4f& v = view_matrices[vs];
            view_mat_uniform->value(0, v);

            if (_model_grid_queries && _model_grid_queries_available) {
                context->begin_query(_time_queries[vs]);
            }
            _model->draw(context);
            //_model->draw_raw(context);
            if (_model_grid_queries && _model_grid_queries_available) {
                context->end_query(_time_queries[vs]);

                if (vs == view_matrices.size() - 1) {
                    _model_grid_queries_available   = false;
                    _model_grid_queries_start_frame = _frame_number;
                    _model_grid_last_query = _time_queries[vs];
                }
            }
        }
    }

    if (_model_grid_last_query && _model_grid_last_query != _time_queries.back()) {
        out() << _model_grid_last_query.get() << " " << _time_queries.back().get() << log::end;
    }
    //if (_model_grid_queries && context->query_result_available(_time_queries.back())) {
    if (_model_grid_last_query && context->query_result_available(_model_grid_last_query)) {
        scm::uint64 frame_delay = _frame_number - _model_grid_queries_start_frame;
        double combined_draw_time = 0.0;
        for (size_t vs = 0; vs < _time_queries.size(); ++vs) {
            context->collect_query_results(_time_queries[vs]);
            combined_draw_time += static_cast<double>(_time_queries[vs]->result()) / 1e3;
        }
        double avg_draw_time = combined_draw_time / static_cast<double>(_time_queries.size());
        if (_enable_output) {
            out() << "grid timer available with " << frame_delay << " frames delay, avg draw time: "
                  << std::fixed << std::setprecision(3) << avg_draw_time << "us." << log::end;
        }
        _model_grid_queries_available = true;
        _model_grid_last_query.reset();
    }

    //context->collect_query_results(_time_query_00);
    //context->collect_query_results(_time_query_01);
    //double draw_time_00 = time::to_milliseconds(time::nanosec(_time_query_00->result()));
    //double draw_time_01 = time::to_milliseconds(time::nanosec(_time_query_01->result()));
    //out() << std::fixed << std::setprecision(3)
    //      << "draw time: " << draw_time_00 << "ms, " << draw_time_01 << "ms, " << log::end;

#endif

#if 0 // check timing of overlapped drawcalls

    const mat4f  model_matrix_00     = make_translation(-0.9f, 0.0f, 0.0f);// * make_scale(vec3f(0.2f));
    const mat4f  model_matrix_01     = make_translation( 0.9f, 0.0f, 0.0f);// * make_scale(vec3f(0.2f));
    const mat4f& view_matrix_00      = _viewer->main_camera().view_matrix() * model_matrix_00;
    const mat4f& view_matrix_01      = _viewer->main_camera().view_matrix() * model_matrix_01;
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    gl::uniform_mat4f_ptr view_mat_uniform = _shader_prog->uniform_mat4f("model_view_matrix");

    {
        context_program_guard       cpg(context);
        context_state_objects_guard csg(context);
        context_texture_units_guard tsg(context);

        _shader_prog->uniform("projection_matrix", proj_matrix);

        context->bind_program(_shader_prog);

        context->bind_texture(_env_map,      _sstate, 0);
        context->bind_texture(_env_map_diff, _sstate, 1);
        context->bind_texture(_env_map,           _sstate, 0);
        context->bind_texture(_env_map_diff,      _sstate, 1);
        context->bind_texture(_model_normal_map,  _sstate, 2);
        context->bind_texture(_model_diffuse_map, _sstate, 3);

        context->set_rasterizer_state(_rstate);
        context->set_blend_state(_bstate);
        context->set_depth_stencil_state(_dstate);

        { // 00 left
            context->begin_query(_time_query_00);
            view_mat_uniform->value(0, view_matrix_00);
            _model->draw_raw(context);
            context->end_query(_time_query_00);
        }

        { // 01 right
            context->begin_query(_time_query_01);
            view_mat_uniform->value(0, view_matrix_01);
            _model->draw_raw(context);
            context->end_query(_time_query_01);
        }
    }

    context->collect_query_results(_time_query_00);
    context->collect_query_results(_time_query_01);
    double draw_time_00 = time::to_milliseconds(time::nanosec(_time_query_00->result()));
    double draw_time_01 = time::to_milliseconds(time::nanosec(_time_query_01->result()));
    out() << std::fixed << std::setprecision(3)
          << "draw time: " << draw_time_00 << "ms, " << draw_time_01 << "ms, " << log::end;

#endif

#if 0 // check frame latency of timer availability
    const mat4f  model_matrix        = make_scale(vec3f(0.2f));
    const mat4f& view_matrix         = _viewer->main_camera().view_matrix() * model_matrix;
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    {
        context_program_guard       cpg(context);
        context_state_objects_guard csg(context);
        context_texture_units_guard tsg(context);

        _shader_prog->uniform("projection_matrix", proj_matrix);
        _shader_prog->uniform("model_view_matrix", view_matrix);

        context->bind_program(_shader_prog);

        context->bind_texture(_env_map,      _sstate, 0);
        context->bind_texture(_env_map_diff, _sstate, 1);
        context->bind_texture(_env_map,           _sstate, 0);
        context->bind_texture(_env_map_diff,      _sstate, 1);
        context->bind_texture(_model_normal_map,  _sstate, 2);
        context->bind_texture(_model_diffuse_map, _sstate, 3);

        context->set_rasterizer_state(_rstate);
        context->set_blend_state(_bstate);
        context->set_depth_stencil_state(_dstate);

        if (_timer_available) {
            context->begin_query(_time_query);
        }
        _model->draw_raw(context);

        if (_timer_available) {
            context->end_query(_time_query);
            _timer_start_frame = _frame_number;
            _timer_available   = false;
        }
    }

    // DO NOT DO THIS!!!!!!!
    //context->collect_query_results(_time_query);

    // DO THIS...
    if (context->query_result_available(_time_query)) {
        scm::uint64 frame_delay = _frame_number - _timer_start_frame;
        out() << "timer available with " << frame_delay << " frames delay.";
        _timer_available = true;
    }
#endif

#if 0 // check frame latency of fence availability
    const mat4f  model_matrix        = make_scale(vec3f(0.2f)) * rotation_matrix;
    const mat4f& view_matrix         = _viewer->main_camera().view_matrix() * model_matrix;
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    {
        context_program_guard       cpg(context);
        context_state_objects_guard csg(context);
        context_texture_units_guard tsg(context);

        _shader_prog->uniform("projection_matrix", proj_matrix);
        _shader_prog->uniform("model_view_matrix", view_matrix);

        context->bind_program(_shader_prog);

        context->bind_texture(_env_map,      _sstate, 0);
        context->bind_texture(_env_map_diff, _sstate, 1);
        context->bind_texture(_env_map,           _sstate, 0);
        context->bind_texture(_env_map_diff,      _sstate, 1);
        context->bind_texture(_model_normal_map,  _sstate, 2);
        context->bind_texture(_model_diffuse_map, _sstate, 3);

        context->set_rasterizer_state(_rstate);
        context->set_blend_state(_bstate);
        context->set_depth_stencil_state(_dstate);

        _model->draw_raw(context);
        if (_fence_available) {
            _fence                 = context->insert_fence_sync();
            _fence_insertion_frame = _frame_number;
            _fence_available       = false;
        }
    }

    // DO NOT DO THIS!!!!!!!
    //if (context->sync_client_wait(_fence) != SYNC_WAIT_CONDITION_SATISFIED) {
    //    err() << "fence wait failed." << log::end;
    //}

    // DO THIS...
    if (context->sync_signal_status(_fence) == SYNC_SIGNALED) {
        scm::uint64 frame_delay = _frame_number - _fence_insertion_frame;
        out() << "fence available with " << frame_delay << " frames delay.";
        _fence_available = true;
    }
#endif
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
    //out() << "application_window::keyboard_input()" << log::end;
    viewer_window::keyboard_input(k, state, mod);
    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_Escape:    close_program();break;
            case Qt::Key_Space:     _animation_enabled = !_animation_enabled;break;
            case Qt::Key_G:         _model_grid_queries = !_model_grid_queries;break;
            case Qt::Key_O:         _enable_output = !_enable_output;break;
            case Qt::Key_S:         reload_shaders(_viewer->device());break;
            default:;
        }
    }
    //switch(k) { // key toggles
    //    default:;
    //}
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
