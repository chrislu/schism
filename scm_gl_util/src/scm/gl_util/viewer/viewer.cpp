
#include "viewer.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>
#include <scm/core/math.h>

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
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/window_management/context.h>
#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/window.h>


namespace {

std::string color_present_vsrc = "\
    #version 330\n\
    \
    uniform mat4 mvp;\
    out vec2 tex_coord;\
    layout(location = 0) in vec3 in_position;\
    layout(location = 2) in vec2 in_texture_coord;\
    \
    void main()\
    {\
        gl_Position = mvp * vec4(in_position, 1.0);\
        tex_coord = in_texture_coord;\
    }\
    ";

std::string color_present_fsrc = "\
    #version 330\n\
    \
    in vec2 tex_coord;\
    uniform sampler2D in_texture;\
    uniform int       in_level;\
    \
    layout(location = 0) out vec4 out_color;\
    void main()\
    {\
        out_color = texelFetch(in_texture, ivec2(gl_FragCoord.xy), in_level).rgba;\
    }\
    ";

} // namespace 

namespace scm {
namespace gl {

viewer::viewer_settings::viewer_settings()
  : _vsync(true)
  , _swap_explicit(false)
  , _clear_color(0.2f, 0.2f, 0.2f)
  , _clear_depth(1.0f)
  , _clear_stencil(0)
  , _show_frame_times(false)
{
}

viewer::viewer_attributes::viewer_attributes()
  : _multi_samples(1)
  , _super_samples(1)
{
}

viewer::render_target::render_target()
  : _viewport_scale(0)
  , _viewport_color_mip_level(0)
{
}

viewer::render_target::~render_target()
{
    _color_present_program.reset();

    _color_buffer.reset();
    _color_buffer_resolved.reset();
    _depth_buffer.reset();

    _framebuffer.reset();
    _framebuffer_resolved.reset();

    _filter_nearest.reset();
    _no_blend.reset();
    _dstate_no_zwrite.reset();
    _cull_back.reset();

    _quad_geom.reset();
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
    try {
        using namespace scm::gl;
        using namespace scm::math;

        wm::display_ptr     default_display(new wm::display(""));
        _window.reset(new wm::window(default_display, parent_wnd, "scm::gl::viewer", vec2i(0, 0), vp_dim, win_fmt));
        _window_context.reset(new wm::context(_window, ctx_attrib));
        _window_context->make_current(_window);

        _device.reset(new render_device());
        _context = _device->main_context();

        initialize_render_target();

        font_face_ptr counter_font(new font_face(_device, "../../../res/fonts/Consola.ttf", 12, 0, font_face::smooth_lcd));
        _text_renderer.reset(new text_renderer(_device));
        _frame_counter_text.reset(new text(_device, counter_font, font_face::style_regular, "sick, sad world..."));
        _frame_counter_text->text_color(math::vec4f(1.0f, 1.0f, 0.0f, 1.0f));
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

    _trackball.dolly(3.0f);
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
    return (_device);
}

const render_context_ptr&
viewer::context() const
{
    return (_context);
}

const viewer::viewer_settings&
viewer::settings() const
{
    return (_settings);
}

viewer::viewer_settings&
viewer::settings()
{
    return (_settings);
}

const wm::window_ptr&
viewer::window() const
{
    return (_window);
}

const wm::context_ptr&
viewer::window_context() const
{
    return (_window_context);
}

const camera&
viewer::main_camera() const
{
    return (_camera);
}

camera&
viewer::main_camera()
{
    return (_camera);
}

const viewport&
viewer::main_viewport() const
{
    return (_viewport);
}

void
viewer::clear_color() const
{
    using namespace scm::math;

    if (   (_attributes._multi_samples > 1)
        || (_attributes._super_samples > 1)) {
        context()->clear_color_buffer(_render_target->_framebuffer, 0, _settings._clear_color);
        context()->clear_color_buffer(_render_target->_framebuffer_resolved, 0, _settings._clear_color);
    }
    else {
        context()->clear_default_color_buffer(FRAMEBUFFER_BACK, _settings._clear_color);
    }
}

void
viewer::clear_depth_stencil() const
{
    using namespace scm::math;

    if (   (_attributes._multi_samples > 1)
        || (_attributes._super_samples > 1)) {
        context()->clear_depth_stencil_buffer(_render_target->_framebuffer, _settings._clear_depth, _settings._clear_stencil);
    }
    else {
        context()->clear_default_depth_stencil_buffer(_settings._clear_depth, _settings._clear_stencil);
    }
}

void
viewer::swap_buffers(int interval)
{
    _window->swap_buffers(interval);
}


void
viewer::render_update_func(const update_func& f)
{
    _update_func = f;
}

void
viewer::render_resize_func(const resize_func& f)
{
    _resize_func = f;
}

void
viewer::render_display_func(const display_func& f)
{
    _display_func = f;
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
viewer::send_render_update()
{
    using namespace scm::math;

    //_device_space_navigator->
    _device_space_navigator->update();

    mat4f view_matrix =   inverse(_device_space_navigator->translation())
                        * inverse(_device_space_navigator->rotation())
                        * _camera.view_matrix();
    _camera.view_matrix(view_matrix);

    if (_update_func) {
        _update_func(device(), context());
    }
}

void
viewer::send_render_display()
{
    using namespace scm::gl;
    using namespace scm::math;

    if (_display_func) {
        
        // clear
        clear_color();
        clear_depth_stencil();

        context_all_guard cguard(context());

        // for AA (multi or super sampling) render to render target
        if (   (_attributes._multi_samples > 1) || (_attributes._super_samples > 1)) {
            context_all_guard fguard(context());

            // set the render target
            context()->set_frame_buffer(_render_target->_framebuffer);
            context()->set_viewport(viewport(vec2ui(0, 0), vec2ui(_viewport._dimensions) * _render_target->_viewport_scale));
            
            // client code
            _display_func(context());

            // resolve framebuffer
            context()->resolve_multi_sample_buffer(_render_target->_framebuffer, _render_target->_framebuffer_resolved);
            if (_attributes._super_samples > 1) {
                context()->generate_mipmaps(_render_target->_color_buffer_resolved);
            }

            // present color buffer
            mat4f   pass_mvp = mat4f::identity();
            ortho_matrix(pass_mvp, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

            _render_target->_color_present_program->uniform("in_texture", 0);
            _render_target->_color_present_program->uniform("in_level", _render_target->_viewport_color_mip_level);
            _render_target->_color_present_program->uniform("mvp", pass_mvp);

            context()->set_default_frame_buffer();
            context()->set_viewport(_viewport);
            
            context()->set_depth_stencil_state(_render_target->_dstate_no_zwrite);
            context()->set_blend_state(_render_target->_no_blend);

            context()->bind_program(_render_target->_color_present_program);
            context()->bind_texture(_render_target->_color_buffer_resolved, _render_target->_filter_nearest, 0);
            _render_target->_quad_geom->draw(context());
        }
        else {
            _display_func(context());
        }

        if (_settings._show_frame_times) {
            mat4f   fs_projection = mat4f::identity();
            ortho_matrix(fs_projection, 0.0f, _viewport._dimensions.x,
                                        0.0f, _viewport._dimensions.y, -1.0f, 1.0f);
            _text_renderer->projection_matrix(fs_projection);

            vec2i text_ur = vec2i(_viewport._dimensions) - _frame_counter_text->text_bounding_box();
            _text_renderer->draw_shadowed(context(), text_ur, _frame_counter_text);
        }
    }
    if (!_settings._swap_explicit) {
        swap_buffers(_settings._vsync ? 1 : 0);
    }

    _frame_time.stop();
    _frame_time.start();

    if (_settings._show_frame_times) {
        if (scm::time::to_milliseconds(_frame_time.accumulated_duration()) > 100.0) {
            std::stringstream   output;
            output.precision(2);
            double frame_time = scm::time::to_milliseconds(_frame_time.average_duration());
            double frame_fps  = 1.0 / scm::time::to_seconds(_frame_time.average_duration());
            output << std::fixed << "frame_time: " << frame_time << "ms "
                                 << "fps: " << frame_fps;

            _frame_counter_text->text_string(output.str());
            if (frame_time > 1000.0 / 50.0) {
                _frame_counter_text->text_color(math::vec4f(1.0f, 0.0f, 0.0f, 1.0f));
            }
            else {
                _frame_counter_text->text_color(math::vec4f(1.0f, 1.0f, 0.0f, 1.0f));
            }
            _frame_time.reset();
        }
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
viewer::send_keyboard_input(int key, bool state)
{
    if (_keyboard_func) {
        _keyboard_func(key, state);
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

math::vec2f
viewer::norm_viewport_coords(const math::vec2i& pos) const
{
    float x     = static_cast<float>(pos.x);
    float y     = static_cast<float>(pos.y);
    float dim_x = static_cast<float>(_viewport._dimensions.x);
    float dim_y = static_cast<float>(_viewport._dimensions.y);

    return (math::vec2f(2.0f * (        x - dim_x * 0.5f) / dim_x,
                        2.0f * (dim_y - y - dim_y * 0.5f) / dim_y));
}

void
viewer::enable_main_manipulator(const bool f)
{
    _trackball_enabled = f;
}

bool
viewer::main_manipulator_enabled() const
{
    return (_trackball_enabled);
}

bool
viewer::initialize_render_target()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    _render_target->_viewport_scale           = static_cast<int>(floor(sqrt(static_cast<double>(_attributes._super_samples))));
    _render_target->_viewport_color_mip_level = floor_log2(static_cast<uint32>(_render_target->_viewport_scale));

     // textures
    _render_target->_color_buffer          = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_RGBA_8, 1, 1, _attributes._multi_samples);
    _render_target->_depth_buffer          = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_D24_S8, 1, 1, _attributes._multi_samples);
    _render_target->_color_buffer_resolved = device()->create_texture_2d(vec2ui(_viewport._dimensions) * _render_target->_viewport_scale, FORMAT_RGBA_8);

    if (   !_render_target->_color_buffer
        || !_render_target->_color_buffer_resolved
        || !_render_target->_depth_buffer) {
        err() << "viewer::initialize_render_target(): error creating textures" << log::end;
        return (false);
    }

    // framebuffers
    _render_target->_framebuffer           = device()->create_frame_buffer();
    _render_target->_framebuffer_resolved  = device()->create_frame_buffer();

    if (   !_render_target->_framebuffer
        || !_render_target->_framebuffer_resolved) {
        err() << "viewer::initialize_render_target(): error creating framebuffers" << log::end;
        return (false);
    }

    _render_target->_framebuffer->attach_color_buffer(0, _render_target->_color_buffer);
    _render_target->_framebuffer->attach_depth_stencil_buffer(_render_target->_depth_buffer);
    _render_target->_framebuffer_resolved->attach_color_buffer(0, _render_target->_color_buffer_resolved);

    // state objects
    _render_target->_filter_nearest   = device()->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _render_target->_no_blend         = device()->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _render_target->_dstate_no_zwrite = device()->create_depth_stencil_state(false, false);
    _render_target->_cull_back        = device()->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW);

    if (   !_render_target->_filter_nearest
        || !_render_target->_no_blend
        || !_render_target->_dstate_no_zwrite
        || !_render_target->_cull_back) {
        err() << "viewer::initialize_render_target(): error creating state objects" << log::end;
        return (false);
    }

    // shader programs
    _render_target->_color_present_program = device()->create_program(list_of(device()->create_shader(STAGE_VERTEX_SHADER, color_present_vsrc))
                                                                             (device()->create_shader(STAGE_FRAGMENT_SHADER, color_present_fsrc)));
    if (   !_render_target->_color_present_program) {
        scm::err() << "viewer::initialize_render_target()): error creating pass through shader program" << log::end;
        return (false);
    }

    _render_target->_quad_geom = make_shared<gl::quad_geometry>(device(), vec2f(0.0f, 0.0f), vec2f(1.0f, 1.0f));

    return (true);
}

} // namespace gl
} // namespace scm
