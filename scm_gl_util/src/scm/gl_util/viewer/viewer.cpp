
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "viewer.h"

#include <iostream>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <FreeImagePlus.h>

#include <scm/log.h>
#include <scm/core/math.h>
#include <scm/core/io/file.h>

#include <scm/input/devices/space_navigator_device.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/frame_buffer_objects.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/math.h>

#include <scm/gl_core/render_device/opengl/util/error_helper.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>
#include <scm/gl_util/primitives/fullscreen_triangle.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_core/window_management/context.h>
#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/window.h>


namespace {

const std::string color_present_vsrc = "                                                \
    #version 330 core                                                                   \n\
                                                                                        \n\
    uniform mat4 mvp;                                                                   \n\
    out vec2 tex_coord;                                                                 \n\
    layout(location = 0) in vec3 in_position;                                           \n\
    layout(location = 2) in vec2 in_texture_coord;                                      \n\
                                                                                        \n\
    void main()                                                                         \n\
    {                                                                                   \n\
        gl_Position = mvp * vec4(in_position, 1.0);                                     \n\
        tex_coord = in_texture_coord;                                                   \n\
    }                                                                                   \n\
    ";

const std::string color_present_fsrc = "                                                \
    #version 330 core                                                                   \n\
                                                                                        \n\
    in vec2 tex_coord;                                                                  \n\
    uniform sampler2D in_texture;                                                       \n\
    uniform int       in_level;                                                         \n\
                                                                                        \n\
    layout(location = 0) out vec4 out_color;                                            \n\
    void main()                                                                         \n\
    {                                                                                   \n\
        out_color = texelFetch(in_texture, ivec2(gl_FragCoord.xy), in_level).rgba;      \n\
    }                                                                                   \n\
    ";

} // namespace 

namespace scm {
namespace gl {

viewer::viewer_settings::viewer_settings()
  : _vsync(true)
  , _vsync_swap_interval(1)
  , _swap_explicit(false)
  , _clear_color(0.2f, 0.2f, 0.2f)
  , _clear_depth(1.0f)
  , _clear_stencil(0)
  , _show_frame_times(false)
  , _full_screen(false)
{
}

viewer::viewer_attributes::viewer_attributes()
  : _post_process_aa(false)
  , _multi_samples(1)
  , _super_samples(1)
{
}

viewer::render_target::render_target()
  : _viewport_scale(1)
  , _viewport_color_mip_level(0)
{
}

viewer::render_target::~render_target()
{
    _color_present_program.reset();
    _post_process_aa_program.reset();

    _color_buffer_aa.reset();
    _depth_buffer_aa.reset();
    _depth_buffer_resolved.reset();
    _color_buffer_resolved.reset();

    _framebuffer_aa.reset();
    _framebuffer_resolved.reset();

    _filter_nearest.reset();
    _filter_linear.reset();
    _no_blend.reset();
    _dstate_no_zwrite.reset();
    _cull_back.reset();

    _fs_geom.reset();
}

viewer::viewer(const math::vec2ui&                  vp_dim,
               const wm::window::handle             parent_wnd,
               const viewer_attributes&             view_attrib,
               const wm::context::attribute_desc&   ctx_attrib,
               const wm::surface::format_desc&      win_fmt)
  : _viewport(math::vec2ui(0, 0), vp_dim)
  , _trackball_enabled(true)
  , _render_target(new render_target())
  , _attributes(view_attrib)
{

    _attributes._multi_samples = math::max(1u, _attributes._multi_samples);
    _attributes._super_samples = math::max(1u, _attributes._super_samples);

    try {
        using namespace scm::gl;
        using namespace scm::math;

        wm::display_ptr     default_display(new wm::display(""));
        _window.reset(new wm::window(default_display, parent_wnd, "scm::gl::viewer", vec2i(0, 0), vp_dim, win_fmt));
        _window_context.reset(new wm::context(_window, ctx_attrib));
        _window_context->make_current(_window);

        glout() << *_window_context << scm::log::end;

        _device.reset(new render_device());
        _context = _device->main_context();

        initialize_shader_includes();
        initialize_render_target();

        font_face_ptr counter_font(new font_face(_device, "../../../res/fonts/Consola.ttf", 12, 0.7f, font_face::smooth_lcd));
        _text_renderer.reset(new text_renderer(_device));
        _frame_counter_text.reset(new text(_device, counter_font, font_face::style_regular, "sick, sad world..."));
        _frame_counter_text->text_color(math::vec4f(1.0f, 1.0f, 0.0f, 1.0f));
        _frame_counter_text->text_outline_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));
        _frame_counter_text->text_kerning(true);

        _device_space_navigator = make_shared<inp::space_navigator>();
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "viewer::viewer(): failed to initialize rendering device and main context ("
            << "evoking error: " << e.what() << ").";
        glerr() << msg.str() << log::end;
        throw(std::runtime_error(msg.str()));
    }
    glout() << *_device << scm::log::end;

    _trackball.dolly(1.5f);
    _camera.view_matrix(_trackball.transform_matrix());
}

viewer::~viewer()
{
    _render_target.reset();
    _text_renderer.reset();
    _frame_counter_text.reset();

    _context.reset();
    _device.reset();

    _window_context.reset();
    _window.reset();

    //_graphics_context.reset();
}


const render_device_ptr&
viewer::device() const
{
    return _device;
}

const render_context_ptr&
viewer::context() const
{
    return _context;
}

const viewer::viewer_settings&
viewer::settings() const
{
    return _settings;
}

viewer::viewer_settings&
viewer::settings()
{
    return _settings;
}

const wm::window_ptr&
viewer::window() const
{
    return _window;
}

const wm::context_ptr&
viewer::window_context() const
{
    return _window_context;
}

const camera&
viewer::main_camera() const
{
    return _camera;
}

camera&
viewer::main_camera()
{
    return _camera;
}

const viewport&
viewer::main_viewport() const
{
    return _viewport;
}

const gl::frame_buffer_ptr&
viewer::main_framebuffer() const
{
    if (_attributes._multi_samples > 1 || _attributes._super_samples > 1) {
        return _render_target->_framebuffer_aa;
    }
    else {
        return _render_target->_framebuffer_resolved;
    }
}

void
viewer::clear_color() const
{
    using namespace scm::math;

    if (_attributes._multi_samples > 1 || _attributes._super_samples > 1) {
        context()->clear_color_buffer(_render_target->_framebuffer_aa, 0, _settings._clear_color);
    }
    else {
        context()->clear_color_buffer(_render_target->_framebuffer_resolved, 0, _settings._clear_color);
    }
}

void
viewer::clear_depth_stencil() const
{
    using namespace scm::math;

    if (_attributes._multi_samples > 1 || _attributes._super_samples > 1) {
        context()->clear_depth_stencil_buffer(_render_target->_framebuffer_aa, _settings._clear_depth, _settings._clear_stencil);
    }
    else {
        context()->clear_depth_stencil_buffer(_render_target->_framebuffer_resolved, _settings._clear_depth, _settings._clear_stencil);
    }
}

float
viewer::frame_time_us() const
{
    return _frame_time_us;
}

void
viewer::swap_buffers(int interval)
{
    _window->swap_buffers(interval);
}

bool
viewer::take_screenshot(const std::string& f) const
{
    if (!_render_target) {
        glerr() << log::error 
                << "viewer::take_screenshot(): only working if using anti aliasing and therefore a texture render target."
                << log::end;
        return false;
    }

    size_t img_size =  _render_target->_color_buffer_resolved->descriptor()._size.x
                     * _render_target->_color_buffer_resolved->descriptor()._size.y
                     * size_of_format(_render_target->_color_buffer_resolved->format());
    shared_ptr<uint8>   data(new uint8[img_size]);

    if (!context()->retrieve_texture_data(_render_target->_color_buffer_resolved, 0, data.get())) {
        glerr() << log::error 
                << "viewer::take_screenshot(): unable to read back texture data." << log::end;
        return false;
    }

#if 0
    io::file_ptr of(new io::file());

    if (!of->open(f, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc, false)) {
        glerr() << log::error 
                << "viewer::take_screenshot(): error opening output file : " << f << log::end;
        return false;
    }
    if (of->write(data.get(), 0, img_size) != img_size) {
        glerr() << log::error 
                << "viewer::take_screenshot(): unable to write out texture data." << log::end;
        return false;
    }
#else
    fipImage of(FIT_BITMAP,
                _render_target->_color_buffer_resolved->descriptor()._size.x,
                _render_target->_color_buffer_resolved->descriptor()._size.y,
                bit_per_pixel(_render_target->_color_buffer_resolved->format()));
    memcpy(of.accessPixels(), data.get(), img_size);
    of.save(f.c_str());
#endif
    return true;
}

void
viewer::render_update_func(const update_func& f)
{
    _pre_frame_update_func = f;
}

void
viewer::render_pre_frame_update_func(const update_func& f)
{
    _pre_frame_update_func = f;
}

void
viewer::render_post_frame_update_func(const update_func& f)
{
    _post_frame_update_func = f;
}

void
viewer::render_resize_func(const resize_func& f)
{
    _resize_func = f;
}

void
viewer::render_display_func(const display_func& f)
{
    _display_scene_func = f;
}

void
viewer::render_display_scene_func(const display_func& f)
{
    _display_scene_func = f;
}

void
viewer::render_display_gui_func(const display_func& f)
{
    _display_gui_func = f;
}

void
viewer::keyboard_input_func(const keyboard_func& f)
{
    _keyboard_func = f;
}

void
viewer::mouse_double_click_func(const mouse_func& f)
{
    _mouse_double_click_func = f;
}

void
viewer::mouse_press_func(const mouse_func& f)
{
    _mouse_press_func = f;
}

void
viewer::mouse_release_func(const mouse_func& f)
{
    _mouse_release_func = f;
}

void
viewer::mouse_move_func(const mouse_func& f)
{
    _mouse_move_func = f;
}

void
viewer::tablet_input_func(const tablet_func& f)
{
    _tablet_func = f;
}

void
viewer::send_render_update()
{
    send_render_pre_frame_update();
}

void
viewer::send_render_pre_frame_update()
{
    using namespace scm::math;

    _device_space_navigator->update(); // update done directly (poll), callback of the device disabled!

    mat4f view_matrix =   inverse(_device_space_navigator->translation())
                        * inverse(_device_space_navigator->rotation())
                        * _camera.view_matrix();
    _camera.view_matrix(view_matrix);
    _trackball.transform_matrix(view_matrix);

    if (_pre_frame_update_func) {
        _pre_frame_update_func(device(), context());
    }
}

void
viewer::send_render_post_frame_update()
{
    if (_post_frame_update_func) {
        _post_frame_update_func(device(), context());
    }
}

void
viewer::send_render_display()
{
    using namespace scm::gl;
    using namespace scm::math;

    _frame_timer.stop();
    _frame_timer.start();

    _frame_time_us = static_cast<float>(_frame_timer.last_time(time::time_io::usec));//static_cast<float>(scm::time::to_microseconds(_frame_timer.last_time()));

    if (_display_scene_func) {

        // clear
        clear_color();
        clear_depth_stencil();

        context_all_guard cguard(context());

        if (_attributes._multi_samples > 1 || _attributes._super_samples > 1) {
            context_all_guard fguard(context());

            // set the render target
            context()->set_frame_buffer(_render_target->_framebuffer_aa);
            context()->set_viewport(viewport(vec2ui(0, 0), vec2ui(_viewport._dimensions) * _render_target->_viewport_scale));
            
            // client code
            _display_scene_func(context());

            // resolve framebuffer
            context()->resolve_multi_sample_buffer(_render_target->_framebuffer_aa, _render_target->_framebuffer_resolved);
            if (_attributes._super_samples > 1) {
                context()->generate_mipmaps(_render_target->_color_buffer_resolved);
            }
        }
        else {
            context_all_guard fguard(context());

            // set the render target
            context()->set_frame_buffer(_render_target->_framebuffer_resolved);
            {
                // client code
                _display_scene_func(context());
            }
        }


        if (_attributes._post_process_aa) {
            context()->set_default_frame_buffer();
            context()->set_viewport(_viewport);
            
            context()->set_depth_stencil_state(_render_target->_dstate_no_zwrite);
            context()->set_blend_state(_render_target->_no_blend);
            context()->set_rasterizer_state(_render_target->_cull_back);

            context()->bind_program(_render_target->_post_process_aa_program);
            context()->bind_texture(_render_target->_color_buffer_resolved, _render_target->_filter_linear, 0);

            _render_target->_fs_geom->draw(context());
        }
        else { // present results
            context()->set_default_frame_buffer();
            context()->set_viewport(_viewport);
            
            context()->set_depth_stencil_state(_render_target->_dstate_no_zwrite);
            context()->set_blend_state(_render_target->_no_blend);
            context()->set_rasterizer_state(_render_target->_cull_back);

            context()->bind_program(_render_target->_color_present_program);
            context()->bind_texture(_render_target->_color_buffer_resolved, _render_target->_filter_nearest, 0);

            _render_target->_fs_geom->draw(context());
        }

        if (_settings._show_frame_times) {
            mat4f   fs_projection = make_ortho_matrix(0.0f, _viewport._dimensions.x,
                                                      0.0f, _viewport._dimensions.y, -1.0f, 1.0f);
            _text_renderer->projection_matrix(fs_projection);

            vec2i text_ur = vec2i(_viewport._dimensions) - _frame_counter_text->text_bounding_box() - vec2i(5, 0);
            _text_renderer->draw_outlined(context(), text_ur, _frame_counter_text);
            //_text_renderer->draw_shadowed(context(), text_ur, _frame_counter_text);
        }
    }

    if (_display_gui_func) {
        _display_gui_func(context());
    }

    if (_settings._show_frame_times) {
        if (_frame_timer.accumulated_time(time::time_io::msec) > 100.0) {
            _frame_timer.update(0);
        //if (scm::time::to_milliseconds(_frame_timer.accumulated_duration()) > 100.0) {
            std::stringstream   output;
            output.precision(2);
            double frame_time = _frame_timer.average_time(time::time_io::msec);//scm::time::to_milliseconds(_frame_timer.average_duration());
            double frame_fps  = 1.0 / _frame_timer.average_time(time::time_io::sec);//scm::time::to_seconds(_frame_timer.average_duration());
            //output << std::fixed << "frame_time: " << frame_time << "ms "
            //                     << "fps: " << frame_fps;
            output << std::fixed << "frame_time: ";
            _frame_timer.report(output);
            output << " fps: " << frame_fps;

            _frame_counter_text->text_string(output.str());
            if (frame_time > 1000.0 / 50.0) {
                //_frame_counter_text->text_color(math::vec4f(1.0f, 0.0f, 0.0f, 1.0f));
                _frame_counter_text->text_outline_color(math::vec4f(1.0f, 0.0f, 0.0f, 1.0f));
            }
            else {
                //_frame_counter_text->text_color(math::vec4f(1.0f, 1.0f, 0.0f, 1.0f));
                _frame_counter_text->text_outline_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));
            }
            _frame_timer.reset();
        }
    }

    if (!_settings._swap_explicit) {
        const int32 swap_interval = math::max(1, _settings._vsync_swap_interval);
        swap_buffers(_settings._vsync ? swap_interval : 0);
    }

    // hack to catch a bug in fraps
    static bool fraps_bug_catched = false;
    if (!fraps_bug_catched) {
        util::gl_error glerror(_context->opengl_api());
        if (glerror) {
            glout() << log::info <<  "viewer::send_render_display(): fraps bug after swap handled (" << glerror.error_string() << ")." << log::end;
        }
        fraps_bug_catched = true;
    }
}

void
viewer::send_render_reshape(int width, int height)
{
    _viewport = viewport(math::vec2ui(0, 0), math::vec2ui(width, height));
    if (_resize_func) {
        _resize_func(device(), context(), width, height);
    }
    // ok i hate this, but we need to recreate our render targets
    if (!initialize_render_target()) {
        glerr() << log::error
                <<  "viewer::send_render_reshape(): unable to initialize render targets." << log::end;
    }
}

void
viewer::send_keyboard_input(int key, bool state, scm::uint32 mod)
{
    if (_keyboard_func) {
        _keyboard_func(key, state, mod);
    }
}

void
viewer::send_mouse_double_click(mouse_button button, int x, int y)
{
    if (_mouse_double_click_func) {
        _mouse_double_click_func(button, x, y);
    }
}

void
viewer::send_mouse_press(mouse_button button, int x, int y)
{
    if (_trackball_enabled) {
        _trackball_start_pos    = norm_viewport_coords(math::vec2i(x, y));
        _trackball_button       = button;
    }

    if (_mouse_press_func) {
        _mouse_press_func(button, x, y);
    }
}

void
viewer::send_mouse_release(mouse_button button, int x, int y)
{
    if (_trackball_enabled) {
        _trackball_button = viewer::no_button;
    }

    if (_mouse_release_func) {
        _mouse_release_func(button, x, y);
    }
}

void
viewer::send_mouse_move(mouse_button button, int x, int y)
{
    if (_trackball_enabled) {
        math::vec2f cur_pos = norm_viewport_coords(math::vec2i(x, y));

        if (_trackball_button == viewer::left_button) {
            _trackball.rotation(_trackball_start_pos.x,
                                _trackball_start_pos.y,
                                cur_pos.x,
                                cur_pos.y);
        }
        if (_trackball_button == viewer::right_button) {
            _trackball.dolly(/*dolly_sens*/1.0f * (cur_pos.y - _trackball_start_pos.y));
        }
        if (_trackball_button == viewer::middle_button) {
            _trackball.translation(cur_pos.x - _trackball_start_pos.x,
                                   cur_pos.y - _trackball_start_pos.y);
        }
        _camera.view_matrix(_trackball.transform_matrix());
        _trackball_start_pos = cur_pos;
    }

    if (_mouse_move_func) {
        _mouse_move_func(button, x, y);
    }
}

void
viewer::send_tablet_input(tablet_device       device,
                          tablet_pointer_type pointer_type,  
                          math::vec3i const&  pos,
                          math::vec2d const&  x_y_hi_res_glob, 
                          math::vec2i const&  x_y_tilt,
                          double              pressure, 
                          double              rotation, 
                          //double              tangential_pressure,
                          scm::int64          unique_id)
{
    if (_tablet_func) {
        _tablet_func(device,
                     pointer_type,
                     pos,
                     x_y_hi_res_glob,
                     x_y_tilt,
                     pressure, 
                     rotation, 
                     //tangential_pressure,
                     unique_id);
    }
}

math::vec2f
viewer::norm_viewport_coords(const math::vec2i& pos) const
{
    float x     = static_cast<float>(pos.x);
    float y     = static_cast<float>(pos.y);
    float dim_x = static_cast<float>(_viewport._dimensions.x);
    float dim_y = static_cast<float>(_viewport._dimensions.y);

    return math::vec2f(2.0f * (        x - dim_x * 0.5f) / dim_x,
                       2.0f * (dim_y - y - dim_y * 0.5f) / dim_y);
}

void
viewer::enable_main_manipulator(const bool f)
{
    _trackball_enabled = f;
}

bool
viewer::main_manipulator_enabled() const
{
    return _trackball_enabled;
}

bool
viewer::initialize_render_target()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    {

        _render_target->_color_present_program = device()->create_program(list_of(device()->create_shader(STAGE_VERTEX_SHADER, color_present_vsrc,   "viewer::color_present_vsrc"))
                                                                                 (device()->create_shader(STAGE_FRAGMENT_SHADER, color_present_fsrc, "viewer::color_present_fsrc")),
                                                                          "viewer::color_present_program");
        if (   !_render_target->_color_present_program) {
            scm::err() << "viewer::initialize_render_target(): error creating pass through shader program" << log::end;
            return false;
        }
        else {
            _render_target->_color_present_program->uniform("mvp",          make_ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f));
            _render_target->_color_present_program->uniform("in_level",     0);
            _render_target->_color_present_program->uniform_sampler("in_texture", 0);
        }
    }

    if (_attributes._post_process_aa) {
        _render_target->_post_process_aa_program = device()->create_program(list_of(device()->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../res/shader/fxaa_nv.glslv"))
                                                                                   (device()->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../res/shader/fxaa_nv.glslf")),
                                                                          "viewer::post_process_aa_program");
        if (   !_render_target->_post_process_aa_program) {
            scm::out() << log::warning
                       << "viewer::initialize_render_target(): error creating post process AA shader program"
                       << "(disabling post process AA support)" << log::end;
            _attributes._post_process_aa = false;
        }
    }

    // render targets
    if (_attributes._multi_samples > 1 || _attributes._super_samples > 1) {
        _render_target->_viewport_scale           = static_cast<int>(floor(sqrt(static_cast<double>(_attributes._super_samples))));
        _render_target->_viewport_color_mip_level = floor_log2(static_cast<uint32>(_render_target->_viewport_scale));

        // set default uniforms
        _render_target->_color_present_program->uniform("mvp",          make_ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f));
        _render_target->_color_present_program->uniform("in_level",     _render_target->_viewport_color_mip_level);
        _render_target->_color_present_program->uniform_sampler("in_texture", 0);

         // textures
        _render_target->_color_buffer_aa = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_RGBA_8, 1, 1, _attributes._multi_samples);
        _render_target->_depth_buffer_aa = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_D24_S8, 1, 1, _attributes._multi_samples);

        if (   !_render_target->_color_buffer_aa
            || !_render_target->_depth_buffer_aa) {
            err() << "viewer::initialize_render_target(): error creating texturesof size: " << vec2ui(_viewport._dimensions) * _render_target->_viewport_scale << log::end;
            return false;
        }

        // framebuffers
        _render_target->_framebuffer_aa = device()->create_frame_buffer();

        if (   !_render_target->_framebuffer_aa) {
            err() << "viewer::initialize_render_target(): error creating aa framebuffer" << log::end;
            return false;
        }

        _render_target->_framebuffer_aa->attach_color_buffer(0, _render_target->_color_buffer_aa);
        _render_target->_framebuffer_aa->attach_depth_stencil_buffer(_render_target->_depth_buffer_aa);
    }

    {
        _render_target->_color_buffer_resolved = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_RGBA_8, _render_target->_viewport_color_mip_level + 1);
        _render_target->_depth_buffer_resolved = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_D24_S8);

        if (   !_render_target->_depth_buffer_resolved
            || !_render_target->_color_buffer_resolved) {
            err() << "viewer::initialize_render_target(): error creating resolve color or depth texture" << log::end;
            return false;
        }

        _render_target->_framebuffer_resolved  = device()->create_frame_buffer();

        if (!_render_target->_framebuffer_resolved) {
            err() << "viewer::initialize_render_target(): error creating resolve framebuffer" << log::end;
            return false;
        }

        _render_target->_framebuffer_resolved->attach_color_buffer(0, _render_target->_color_buffer_resolved);
        _render_target->_framebuffer_resolved->attach_depth_stencil_buffer(_render_target->_depth_buffer_resolved);

        if (_attributes._post_process_aa) {
            vec2f vp_size = vec2f(_viewport._dimensions);
            _render_target->_post_process_aa_program->uniform("mvp",            make_ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f));
            //_render_target->_post_process_aa_program->uniform("in_vp_size",     vp_size);
            _render_target->_post_process_aa_program->uniform("in_vp_size_rec", vec2f(1.0) / vp_size);
            _render_target->_post_process_aa_program->uniform_sampler("in_texture", 0);

        }

        // state objects
        _render_target->_filter_nearest   = device()->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
        _render_target->_filter_linear    = device()->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);
        _render_target->_no_blend         = device()->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
        _render_target->_dstate_no_zwrite = device()->create_depth_stencil_state(false, false);
        _render_target->_cull_back        = device()->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW);

        if (   !_render_target->_filter_nearest
            || !_render_target->_no_blend
            || !_render_target->_dstate_no_zwrite
            || !_render_target->_cull_back) {
            err() << "viewer::initialize_render_target(): error creating state objects" << log::end;
            return false;
        }

        _render_target->_fs_geom.reset(new gl::fullscreen_triangle(device()));
    }

    return true;
}

bool
viewer::initialize_shader_includes()
{
    //if (!device()->add_include_string(camera_block_include_path, camera_block_include_src)) {
    //    scm::err() << "viewer::initialize_shader_includes(): error adding camera block include string." << log::end;
    //    return false;
    //}
    // just a test
    //if (!device()->add_include_string("/scm/gl_util/viewer/camera_block.glsl", camera_block_include_src)) {
    //    scm::err() << "viewer::initialize_shader_included(): error adding camera block include string." << log::end;
    //    return false;
    //}

    return true;
}

} // namespace gl
} // namespace scm
