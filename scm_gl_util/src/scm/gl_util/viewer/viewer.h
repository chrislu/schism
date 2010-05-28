
#ifndef SCM_GL_UTIL_VIEWER_H_INCLUDED
#define SCM_GL_UTIL_VIEWER_H_INCLUDED

#include <boost/function.hpp>

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/frame_buffer_objects/viewport.h>

#include <scm/gl_util/manipulators/trackball_manipulator.h>
#include <scm/gl_util/render_context/render_context_fwd.h>
#include <scm/gl_util/render_context/context_format.h>
#include <scm/gl_util/render_context/window_context.h>
#include <scm/gl_util/viewer/camera.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) viewer
{
public:
    enum mouse_button {
        no_button       = 0x00,
        left_button     = 0x01,
        middle_button,
        right_button
    }; // enum mouse_button

    struct viewer_settings {
        viewer_settings();

        bool        _vsync;
        bool        _swap_explicit;
        math::vec3f _clear_color;
        float       _clear_depth;
        unsigned    _clear_stencil;
    }; // struct viewer_settings

    typedef boost::function<void (const render_device_ptr&,
                                  const render_context_ptr&)>   update_func;
    typedef boost::function<void (const render_device_ptr&,
                                  const render_context_ptr&,
                                  int, int)>                    resize_func;
    typedef boost::function<void (const render_context_ptr&)>   display_func;

    typedef boost::function<void (int)>                         keyboard_func;
    typedef boost::function<void (mouse_button, int, int)>      mouse_func;

public:
    viewer(const math::vec2ui&              vp_dim,
           const window_context::wnd_handle wnd,
           const context_format&            format = context_format::default_format());
    virtual~viewer();

    const render_device_ptr&        device() const;
    const render_context_ptr&       context() const;

    const window_context&           graphics_context() const;

    const camera&                   main_camera() const;
    camera&                         main_camera();

    void                            clear_color() const;
    void                            clear_depth_stencil() const;

    void                            swap_buffers(int interval = 0);
    
    // callbacks
    void                            render_update_func(const update_func& f);
    void                            render_resize_func(const resize_func& f);
    void                            render_display_func(const display_func& f);

    void                            keyboard_input_func(const keyboard_func& f);
    void                            mouse_double_click_func(const mouse_func& f);
    void                            mouse_press_func(const mouse_func& f);
    void                            mouse_release_func(const mouse_func& f);
    void                            mouse_move_func(const mouse_func& f);

    // callback invoke (to be hidden somewhere else)
    void                            send_render_update();
    void                            send_render_display();
    void                            send_render_reshape(int width, int height);

    void                            send_keyboard_input(int key);
    void                            send_mouse_double_click(mouse_button button, int x, int y);
    void                            send_mouse_press(mouse_button button, int x, int y);
    void                            send_mouse_release(mouse_button button, int x, int y);
    void                            send_mouse_move(mouse_button button, int x, int y);

protected:
    math::vec2f                     norm_viewport_coords(const math::vec2ui& pos) const;

protected:
    // framestamp
    // stats (frametime etc.)

    viewer_settings                 _settings;

    camera                          _camera;
    viewport                        _viewport;

    render_device_ptr               _device;
    render_context_ptr              _context;

    shared_ptr<window_context>      _graphics_context;

    trackball_manipulator           _trackball;
    math::vec2f                     _trackball_start_pos;
    mouse_button                    _trackball_button;

    // callbacks
    update_func                     _update_func;
    resize_func                     _resize_func;
    display_func                    _display_func;

    keyboard_func                   _keyboard_func;
    mouse_func                      _mouse_double_click_func;
    mouse_func                      _mouse_press_func;
    mouse_func                      _mouse_release_func;
    mouse_func                      _mouse_move_func;
    
}; // class viewer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_VIEWER_H_INCLUDED
