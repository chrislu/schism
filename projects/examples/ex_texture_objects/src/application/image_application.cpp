
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "image_application.h"

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

#include <scm/gl_util/data/imaging/texture_image_data.h>
#include <scm/gl_util/data/imaging/texture_loader.h>
#include <scm/gl_util/data/imaging/texture_loader_dds.h>
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

    //_sstate_linear  = device->create_sampler_state(FILTER_MIN_MAG_MIP_LINEAR, WRAP_CLAMP_TO_EDGE);
    _sstate_linear  = device->create_sampler_state(FILTER_ANISOTROPIC, WRAP_CLAMP_TO_EDGE, 16u);
    _sstate_nearest = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);

    if (   !_bstate_off
        || !_dstate_less
        || !_rstate_cback
        || !_sstate_linear
        || !_sstate_nearest) {
        err() << "application_window::initialize(): error creating state objects" << log::end;
        return false;
    }

    // introduce the camera block
    camera_uniform_block::add_block_include_string(device);
    _camera_block.reset(new gl::camera_uniform_block(device));

    { // this way for texture
        texture_loader tl;
        //texture_loader_dds tldds;
        //_texture = tl.load_texture_2d(*device, "e:/data/image/9583_HE_01_01.tif", true, true);
        //_texture = tl.load_texture_2d(*device, "e:/data/image/checkerboard.png", true, true);
        //_texture = tl.load_texture_2d(*device, "e:/data/textures/misc/DH217SN.hdr", true, false);

        _texture = tl.load_texture_2d(*device, "../../../res/textures/0001MM_diff.jpg", true, false);

        if (0) { // test texture readback
            size_t img_size =  _texture->descriptor()._size.x
                             * _texture->descriptor()._size.y
                             * size_of_format(_texture->format());
            shared_ptr<uint8>   data(new uint8[img_size]);

            if (!device->main_context()->retrieve_texture_data(_texture, 0, data.get())) {
                err() << "unable to read back texture data." << log::end;
                return false;
            }
            io::file_ptr f(new io::file());

            f->open("temp_img_out.raw", std::ios_base::out | std::ios_base::binary, false);
            if (f->write(data.get(), 0, img_size) != img_size) {
                err() << "unable to write out texture data." << log::end;
                return false;
            }
        }

        if (!_texture) {
            err() << "application_window::init_renderer(): error loading texture image." << log::end;
            return false;
        }

        _texture_uint_view = device->create_texture_2d(_texture,
                                                       gl::FORMAT_RGB_8UI,
                                                       math::vec2ui(3, 4),//_texture->descriptor()._mip_levels),
                                                       math::vec2ui(0, 1));

        if (!_texture_uint_view) {
            err() << "application_window::init_renderer(): error creating texture view." << log::end;
            return false;
        }

        _texture_resident_handle_lin = device->create_resident_handle(_texture,  _sstate_linear);
        _texture_resident_handle_near = device->create_resident_handle(_texture, _sstate_nearest);
        if (   !_texture_resident_handle_lin
            || !_texture_resident_handle_near) {
            err() << "application_window::init_renderer(): error creating texture resident handles." << log::end;
            return false;
        }
    }


    // load shader program
    _shader_prog = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/shaders/texture_program.glslv"))
                                                 (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/shaders/texture_program.glslf")),
                                            "application_window::shader_prog");

    if (!_shader_prog) {
        err() << "application_window::init_renderer(): error creating vtexture_program" << log::end;
        return false;
    }
    // default uniform values
    _shader_prog->uniform_buffer("camera_matrices", 0);

    // model geometry
    try {
        _model_geometry = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/box.obj");
        //_model_geometry = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/quad.obj");
        //_model_geometry = make_shared<wavefront_obj_geometry>(device, "../../../res/geometry/guardian.obj");
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
    _sstate_linear.reset();
    
    _camera_block.reset();
    
    _texture_resident_handle_lin.reset();
    _texture_resident_handle_near.reset();
    _texture_uint_view.reset();
    _texture.reset();

    _model_geometry.reset();
    _shader_prog.reset();
}

void
application_window::update(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context)
{
    _camera_block->update(context, _viewer->main_camera());
}

void
application_window::display(const gl::render_context_ptr& context)
{
    using namespace scm::gl;
    using namespace scm::math;

    { // image rendering pass
        context_program_guard           cpg(context);
        context_state_objects_guard     csg(context);
        context_texture_units_guard     ctg(context);
        context_uniform_buffer_guard    cug(context);
        context_image_units_guard       cig(context);


        context->bind_uniform_buffer(_camera_block->block().block_buffer(), 0);
        context->bind_program(_shader_prog);

        context->set_rasterizer_state(_rstate_cback);
        context->set_depth_stencil_state(_dstate_less);
        context->set_blend_state(_bstate_off);

        { // texture
            context->bind_texture(_texture,           _sstate_linear,  0);
            context->bind_texture(_texture_uint_view, _sstate_nearest, 1);

            vec2ui tex_handle_lin  = vec2ui(static_cast<uint32>(_texture_resident_handle_lin->native_handle() & 0x00000000ffffffffull),
                                            static_cast<uint32>(_texture_resident_handle_lin->native_handle() >> 32ull));
            vec2ui tex_handle_near = vec2ui(static_cast<uint32>(_texture_resident_handle_near->native_handle() & 0x00000000ffffffffull),
                                            static_cast<uint32>(_texture_resident_handle_near->native_handle() >> 32ull));
            _shader_prog->uniform("tex_color_resident_lin",  tex_handle_lin);
            _shader_prog->uniform("tex_color_resident_near", tex_handle_near);
        }

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
application_window::keyboard_input(int k, bool state)
{
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
