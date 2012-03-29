
#include "image_application.h"

#include <exception>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>

#include <boost/bind.hpp>
#include <boost/assign/list_of.hpp>

#include <QtGui/QApplication>
#include <QtGui/QCloseEvent>
#include <QtGui/QFileDialog>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QMouseEvent>
#include <QtGui/QVBoxLayout>

#include <scm/log.h>
#include <scm/core/math.h>
#include <scm/core/io/tools.h>

#include <scm/gl_core.h>
#include <scm/gl_core/math.h>
#include <scm/gl_core/primitives.h>

#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/quad.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>

#include <renderer/readback_benchmark.h>

#include <gui_support/viewer_widget.h>

#include <application/draw_helpers.h>

namespace {

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
namespace data {

application_window::application_window(const math::vec2ui&                    vp_size,
                                       const gl::viewer::viewer_attributes&   view_attrib,
                                       const gl::wm::context::attribute_desc& ctx_attrib,
                                       const gl::wm::surface::format_desc&    win_fmt)
  : gl::gui::viewer_window(vp_size, view_attrib, ctx_attrib, win_fmt)
  , _viewport_size(vp_size)
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
    std::cout << "application_window::~application_window(): bye, bye..." << std::endl;
}

bool
application_window::init_renderer()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    _viewer->settings()._clear_color      = vec4f(0.15f, 0.15f, 0.15f, 1.0f);
    _viewer->settings()._vsync            = false;
    _viewer->settings()._show_frame_times = true;

    // context
    const render_device_ptr& device = _viewer->device();
    device->main_context()->register_debug_callback(make_shared<scm_debug_output>());

    _no_blend         = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate_less      = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _dstate_no_zwrite = device->create_depth_stencil_state(false, false);
    _cull_back        = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW);
    _cull_back_ms     = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    if (   !_no_blend
        || !_dstate_less
        || !_dstate_no_zwrite
        || !_cull_back
        || !_cull_back_ms) {
        err() << "application_window::initialize(): error creating state objects" << log::end;
        return false;
    }

    try {
        //std::string model        = "e:/data/geometry/happy_budda.obj";
        std::string model        = "../../../res/geometry/guardian.obj";

        _readback_benchmark.reset(new readback_benchmark(device,
                                                         _viewer->window_context(),
                                                         _viewer->window(),
                                                         _viewport_size,
                                                         model));
                                                         //"e:/working_copies/schism_x64/resources/objects/01_renault_clio_scaled.obj"));
                                                         //"../../../res/geometry/guardian.obj"));
                                                         //"../../../res/geometry/box.obj"));
    }
    catch (const std::exception& e) {
        std::stringstream msg;
        msg << "application_window::init_renderer(): unable to initialize the vtexture render system (" 
            << "evoking error: " << e.what() << ").";
        err() << msg.str() << log::end;
        return false;
    }

    try {
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "application_window::init_renderer(): unable to initialize the render system (" 
            << "evoking error: " << e.what() << ").";
        err() << msg.str() << log::end;
        return false;
    }

    return true;
}

void
application_window::shutdown()
{
    std::cout << "application_window::shutdown(): bye, bye..." << std::endl;

    _no_blend.reset();
    _dstate_less.reset();
    _dstate_no_zwrite.reset();
    _cull_back.reset();
    _cull_back_ms.reset();
    
    _readback_benchmark.reset();
}

void
application_window::update(const gl::render_device_ptr&  device,
                           const gl::render_context_ptr& context)
{
    _readback_benchmark->update(context, _viewer->main_camera());
}

void
application_window::display(const gl::render_context_ptr& context)
{
    using namespace scm::gl;
    using namespace scm::math;

    const mat4f& view_matrix         = _viewer->main_camera().view_matrix();
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    { // image rendering pass
        context_state_objects_guard csg(context);
        context_texture_units_guard tug(context);

        context->set_depth_stencil_state(_dstate_less);
        context->set_blend_state(_no_blend);
        context->set_rasterizer_state(_cull_back_ms);

        _readback_benchmark->draw(context);
    }

    { // overlays
    }
}

void
application_window::reshape(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context,
                           int w, int h)
{
    std::cout << "application_window::reshape(): width = " << w << " height = " << h << std::endl;

    using namespace scm::gl;
    using namespace scm::math;

    viewer_window::reshape(device, context, w, h);
}


void
application_window::keyboard_input(int k, bool state, scm::uint32 mod)
{
    viewer_window::keyboard_input(k, state, mod);
    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_Escape:    close_program();break;
            case Qt::Key_C:         _readback_benchmark->capture_enabled(!_readback_benchmark->capture_enabled());break;
            default:;
        }
    }
    //switch(k) { // key toggles
    //    //default:;
    //}
}

void
application_window::mouse_double_click(gl::viewer::mouse_button b, int x, int y)
{
    using namespace scm::math;
    using namespace scm::gl;
}

void
application_window::mouse_press(gl::viewer::mouse_button b, int x, int y)
{
}

void
application_window::mouse_release(gl::viewer::mouse_button b, int x, int y)
{
}

void
application_window::mouse_move(gl::viewer::mouse_button b, int x, int y)
{
}

} // namespace data
} // namespace scm
