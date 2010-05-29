
#include "viewer.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <scm/log.h>
#include <scm/core/math.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>

#include <scm/gl_core/render_device/opengl/util/error_helper.h>

#include <scm/gl_util/render_context/window_context_win32.h>

namespace scm {
namespace gl {


viewer::viewer_settings::viewer_settings()
  : _vsync(true),
    _swap_explicit(false),
    _clear_color(0.2f, 0.2f, 0.2f),
    _clear_depth(1.0f),
    _clear_stencil(0)
{
}

viewer::viewer(const math::vec2ui&              vp_dim,
               const window_context::wnd_handle wnd,
               const context_format&            format)
  : _viewport(math::vec2ui(0, 0), vp_dim)
{

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    _graphics_context.reset(new window_context_win32());
#else // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#error "not yet implemented"
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

    if (!_graphics_context->setup(wnd, format)) {
        std::stringstream msg;
        msg << "viewer::viewer(): failed to initialize window rendering context";
        glerr() << msg.str() << log::end;
        throw(std::runtime_error(msg.str()));
    }

    try {
        _device.reset(new render_device());
        _context = _device->main_context();
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
    _camera.view(_trackball.transform_matrix());
}

viewer::~viewer()
{
    _context.reset();
    _device.reset();
    _graphics_context.reset();
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

const window_context&
viewer::graphics_context() const
{
    assert(_graphics_context);
    return (*_graphics_context);
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

void
viewer::clear_color() const
{
    _context->clear_default_color_buffer(FRAMEBUFFER_BACK, _settings._clear_color);
}

void
viewer::clear_depth_stencil() const
{
    _context->clear_default_depth_stencil_buffer(_settings._clear_depth, _settings._clear_stencil);
}

void
viewer::swap_buffers(int interval)
{
    assert(_graphics_context);
    _graphics_context->swap_buffers(interval);
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
    if (_update_func) {
        _update_func(device(), context());
    }
}

void
viewer::send_render_display()
{
    if (_display_func) {
        _display_func(context());
    }
    if (!_settings._swap_explicit) {
        swap_buffers(_settings._vsync ? 1 : 0);
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
}

void
viewer::send_keyboard_input(int key)
{
    if (_keyboard_func) {
        _keyboard_func(key);
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
    _trackball_start_pos    = norm_viewport_coords(math::vec2i(x, y));
    _trackball_button       = button;

    if (_mouse_press_func) {
        _mouse_press_func(button, x, y);
    }
}

void
viewer::send_mouse_release(mouse_button button, int x, int y)
{
    _trackball_button = viewer::no_button;

    if (_mouse_release_func) {
        _mouse_release_func(button, x, y);
    }
}

void
viewer::send_mouse_move(mouse_button button, int x, int y)
{
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
    _camera.view(_trackball.transform_matrix());
    _trackball_start_pos = cur_pos;

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


} // namespace gl
} // namespace scm
