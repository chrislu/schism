
#ifndef SCM_GL_UTIL_VIEWER_H_INCLUDED
#define SCM_GL_UTIL_VIEWER_H_INCLUDED

#include <boost/function.hpp>

#include <scm/core/math.h>
#include <scm/core/time/accumulate_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/input/devices/devices_fwd.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/frame_buffer_objects/viewport.h>

#include <scm/gl_util/manipulators/trackball_manipulator.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/window_management/wm_fwd.h>
#include <scm/gl_util/window_management/surface.h>
#include <scm/gl_util/window_management/window.h>
#include <scm/gl_util/window_management/context.h>

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

    struct __scm_export(gl_util) viewer_attributes {
        viewer_attributes();

        bool            _post_process_aa;
        unsigned        _multi_samples;
        unsigned        _super_samples;
    }; // struct viewer_attributes

    struct __scm_export(gl_util) viewer_settings {
        viewer_settings();

        bool        _vsync;
        bool        _swap_explicit;
        math::vec3f _clear_color;
        float       _clear_depth;
        unsigned    _clear_stencil;
        bool        _show_frame_times;
    }; // struct viewer_settings

    typedef scm::time::accumulate_timer<scm::time::high_res_timer>  accum_timer_type;

    typedef boost::function<void (const render_device_ptr&,
                                  const render_context_ptr&)>   update_func;
    typedef boost::function<void (const render_device_ptr&,
                                  const render_context_ptr&,
                                  int, int)>                    resize_func;
    typedef boost::function<void (const render_context_ptr&)>   display_func;

    typedef boost::function<void (int, bool)>                   keyboard_func;
    typedef boost::function<void (mouse_button, int, int)>      mouse_func;

public:
    viewer(const math::vec2ui&                  vp_dim,
           const wm::window::handle             parent_wnd = 0,
           const viewer_attributes&             view_attrib = viewer_attributes(),
           const wm::context::attribute_desc&   ctx_attrib = wm::context::default_attributes(),
           const wm::surface::format_desc&      win_fmt = wm::surface::default_format());
    virtual~viewer();

    const render_device_ptr&        device() const;
    const render_context_ptr&       context() const;

    const viewer_settings&          settings() const;
    viewer_settings&                settings();

    //const window_context&           graphics_context() const;
    const wm::window_ptr&           window() const;
    const wm::context_ptr&          window_context() const;

    const camera&                   main_camera() const;
    camera&                         main_camera();

    const viewport&                 main_viewport() const;

    void                            enable_main_manipulator(const bool f);
    bool                            main_manipulator_enabled() const;

    math::vec2f                     norm_viewport_coords(const math::vec2i& pos) const;

    void                            clear_color() const;
    void                            clear_depth_stencil() const;

    void                            swap_buffers(int interval = 0);

    bool                            take_screenshot(const std::string& f) const;
    
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

    void                            send_keyboard_input(int key, bool state);
    void                            send_mouse_double_click(mouse_button button, int x, int y);
    void                            send_mouse_press(mouse_button button, int x, int y);
    void                            send_mouse_release(mouse_button button, int x, int y);
    void                            send_mouse_move(mouse_button button, int x, int y);

protected:
    bool                            initialize_render_target();
    bool                            initialize_shader_includes();

protected:
    // framestamp
    // stats (frametime etc.)

    viewer_settings                 _settings;

    camera                          _camera;
    viewport                        _viewport;

    render_device_ptr               _device;
    render_context_ptr              _context;

    wm::window_ptr                  _window;
    wm::context_ptr                 _window_context;

    viewer_attributes               _attributes;

    struct render_target {
        render_target();
        ~render_target();
        gl::program_ptr                 _color_present_program;
        gl::program_ptr                 _post_process_aa_program;
        int                             _viewport_scale;
        int                             _viewport_color_mip_level;

        // frame buffer
        gl::texture_2d_ptr              _color_buffer_aa;
        gl::texture_2d_ptr              _depth_buffer_aa;
        gl::texture_2d_ptr              _color_buffer_resolved;
        gl::texture_2d_ptr              _depth_buffer_resolved;

        gl::frame_buffer_ptr            _framebuffer_aa;
        gl::frame_buffer_ptr            _framebuffer_resolved;
   
        // state objects
        gl::sampler_state_ptr           _filter_nearest;
        gl::sampler_state_ptr           _filter_linear;
        gl::blend_state_ptr             _no_blend;
        gl::depth_stencil_state_ptr     _dstate_no_zwrite;
        gl::rasterizer_state_ptr        _cull_back;

        // fullscreen quad geometry
        shared_ptr<gl::quad_geometry>   _quad_geom;

    };
    shared_ptr<render_target>       _render_target;

    accum_timer_type                _frame_time;
    gl::text_renderer_ptr           _text_renderer;
    gl::text_ptr                    _frame_counter_text;

    trackball_manipulator           _trackball;
    math::vec2f                     _trackball_start_pos;
    mouse_button                    _trackball_button;
    bool                            _trackball_enabled;

    inp::space_navigator_ptr        _device_space_navigator;

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
