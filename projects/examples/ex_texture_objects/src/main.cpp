
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <string>
#include <sstream>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <QtCore/QSettings>
#include <QtGui/QApplication>
#include <QtGui/QPlastiqueStyle>
#include <QtGui/QCleanlooksStyle>
#include <QtGui/QWindowsXPStyle>
#include <QtGui/QWindowsVistaStyle>

#include <scm/core.h>
#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/platform/platform.h>

//#include <scm/gl_util/render_context/context_format.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_util/viewer/viewer.h>
#include <scm/gl_core/window_management/context.h>
#include <scm/gl_core/window_management/surface.h>

#include <application/image_application.h>

namespace  {

scm::math::vec2ui viewport_size;
bool              viewport_fullscreen;
unsigned          multi_samples;

scm::size_t hdc;
scm::size_t atlas_s;

static const std::string    scm_application_name = "schism example: texture objects";

} // namespace

static bool initialize_cmd_line(scm::core& c)
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    options_description  cmd_options("program options");

    cmd_options.add_options()
        ("?",                                                                           "show this help message")
        ("width,w",         value<unsigned>(&viewport_size.x)->default_value(1024),    "output width")
        ("height,h",        value<unsigned>(&viewport_size.y)->default_value(640),     "output height")
        ("multi_samples,s", value<unsigned>(&multi_samples)->default_value(1),         "multi samples (AA mode)")
        ("fullscreen,f",    value<bool>(&viewport_fullscreen)->zero_tokens(),          "run in fullscreen mode");

    c.add_command_line_options(cmd_options, scm_application_name);
    c.command_line_positions().add("width",         1)  // max occurances 1
                              .add("height",        1)  // max occurances 1
                              .add("multi_samples", 1); // max occurances 1

    return (true);
}

static void init_module()
{
    scm::module::initializer::add_pre_core_init_function(initialize_cmd_line);
}

static scm::module::static_initializer  static_initialize(init_module);

int main(int argc, char **argv)
{
    // some standard things to turn off
    std::ios_base::sync_with_stdio(false);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    using namespace scm;
    using namespace scm::gl;

    shared_ptr<core> scm_core(new core(argc, argv));

    ///////////////////////////////////////////////////////////////////////////////////////////////
    QApplication     app(argc, argv);

    viewer::viewer_attributes   viewer_attribs;
    wm::surface::format_desc    window_format(FORMAT_RGBA_8,
                                              FORMAT_D24_S8,
                                              true /*double_buffer*/,
                                              false /*quad_buffer*/);
    wm::context::attribute_desc context_attribs(4, //SCM_GL_CORE_BASE_OPENGL_VERSION / 100,
                                                3, //SCM_GL_CORE_BASE_OPENGL_VERSION / 10 % 10,
                                                false /*compatibility*/,
                                                false /*debug*/,
                                                false /*forward*/,
                                                false /*es profile*/);

    viewer_attribs._multi_samples = multi_samples;
    viewer_attribs._super_samples = 1;

   // data::application_window* app_window(new data::application_window(math::vec2ui(width, height), viewer_attribs, context_attribs, window_format));
    data::application_window* app_window(new data::application_window(viewport_size, viewer_attribs, context_attribs, window_format));

    app_window->setWindowTitle(scm_application_name.c_str());
    app_window->show();

    return app.exec();
}
