
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "application.h"

#include <exception>
#include <stdexcept>
#include <fstream>
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
#include <scm/core/io/file.h>

#include <scm/gl_core.h>
#include <scm/gl_core/math.h>
#include <scm/gl_core/primitives.h>

#include <scm/gl_util/imaging/texture_image_data.h>
#include <scm/gl_util/imaging/texture_loader.h>
#include <scm/gl_util/imaging/texture_loader_dds.h>
#include <scm/gl_util/primitives/wavefront_obj.h>
#include <scm/gl_util/utilities/geometry_highlight.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/viewer/camera_uniform_block.h>

#include <gui_support/viewer_widget.h>

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
    // file menu
    QMenu*          file_menu       = new QMenu("Renderer", this);

    _main_menubar->addMenu(file_menu);

    if (!init_renderer()) {
        std::stringstream msg;
        msg << "application_window::application_window(): error initializing rendering stystem.";
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
    using boost::assign::list_of;

    _viewer->settings()._clear_color = vec4f(0.15f, 0.15f, 0.15f, 1.0f);

    // context
    const render_device_ptr& device = _viewer->device();
    device->main_context()->register_debug_callback(make_shared<scm_debug_output>());

    _bstate_off     = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate_less    = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _rstate_cback   = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    if (   !_bstate_off
        || !_dstate_less
        || !_rstate_cback) {
        err() << "application_window::initialize(): error creating state objects" << log::end;
        return false;
    }

    // introduce the camera block
    camera_uniform_block::add_block_include_string(device);
    _camera_block.reset(new gl::camera_uniform_block(device));

    _atomic_counter = device->create_buffer(BIND_ATOMIC_COUNTER_BUFFER, USAGE_STATIC_DRAW, sizeof(unsigned int));
    if (!_atomic_counter) {
        err() << "application_window::init_renderer(): error creating atomic_counter buffer." << log::end;
        return false;
    }

    // load shader program
    _atomic_test_prog = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/shaders/atomic_test.glslv"))
                                                      (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/shaders/atomic_test.glslf")),
                                            "application_window::atomic_test_prog");

    if (!_atomic_test_prog) {
        err() << "application_window::init_renderer(): error creating vtexture_program" << log::end;
        return false;
    }
    // default uniform values
    _atomic_test_prog->uniform_buffer("camera_matrices", 0);

    // model geometry
    try {
        _model_geometry = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/box.obj");
    }
    catch(const std::exception& e) {
        err() << "application_window::init_renderer(): " <<  e.what() << log::end;
        return false;
    }


    return true;
}

void
application_window::shutdown()
{
    std::cout << "application_window::shutdown(): bye, bye..." << std::endl;

    _bstate_off.reset();
    _dstate_less.reset();
    _rstate_cback.reset();
    
    _camera_block.reset();
    
    _model_geometry.reset();
    _atomic_test_prog.reset();
}

void
application_window::update(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context)
{
    _camera_block->update(context, _viewer->main_camera());

    unsigned* d = reinterpret_cast<unsigned*>(context->map_buffer(_atomic_counter, gl::ACCESS_WRITE_INVALIDATE_BUFFER));
    if (d) {
        *d = 0u;
    }
    context->unmap_buffer(_atomic_counter);
}

void
application_window::display(const gl::render_context_ptr& context)
{
    using namespace scm::gl;
    using namespace scm::math;

    { // image rendering pass
        context_program_guard               cpg(context);
        context_state_objects_guard         csg(context);
        context_texture_units_guard         ctg(context);
        context_uniform_buffer_guard        cug(context);
        context_atomic_counter_buffer_guard acg(context);
        context_image_units_guard           cig(context);

        _atomic_test_prog->uniform("screen_res", vec2f(_viewport_size));

        context->bind_atomic_counter_buffer(_atomic_counter, 0);
        context->bind_uniform_buffer(_camera_block->block().block_buffer(), 0);
        context->bind_program(_atomic_test_prog);

        context->set_rasterizer_state(_rstate_cback);
        context->set_depth_stencil_state(_dstate_less);
        context->set_blend_state(_bstate_off);

        _model_geometry->draw_raw(context, geometry::MODE_SOLID);
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
    //_viewer->main_camera().projection_ortho(-1.0f, 1.0f, -1.0f, 1.0f, -10.0f, 10.0f);
    _viewer->main_camera().projection_perspective(60.f, float(w)/float(h), 0.001f, 10.0f);
}


void
application_window::keyboard_input(int k, bool state, scm::uint32 mod)
{
    viewer_window::keyboard_input(k, state, mod);
    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_Escape:    close_program();break;
            case Qt::Key_S:         _viewer->take_screenshot("screen.tif");break;
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
