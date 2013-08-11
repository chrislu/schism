
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <string>
#include <sstream>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/program_options.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <QtGui/QApplication>

#include <scm/core.h>
#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/platform/platform.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/data_formats.h>

#include <scm/gl_util/viewer/viewer.h>
#include <scm/gl_core/window_management/context.h>
#include <scm/gl_core/window_management/surface.h>

#include <application/application.h>

namespace  {

scm::math::vec2ui viewport_size;
bool              viewport_fullscreen;
unsigned          multi_samples;

std::string       height_field_file;

} // namespace

static const std::string    scm_application_name = "schism example: height field tessellation";

static bool initialize_cmd_line(scm::core& c)
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    options_description  cmd_options("program options");

    cmd_options.add_options()
        ("?",                                                                           "show this help message")
        ("i",                value<std::string>(&height_field_file),                    "input file")
        ("width,w",          value<unsigned>(&viewport_size.x)->default_value(1024),    "output width")
        ("height,h",         value<unsigned>(&viewport_size.y)->default_value(640),     "output height")
        ("multi_samples,s",  value<unsigned>(&multi_samples)->default_value(1),         "multi samples (AA mode)")
        ("fullscreen,f",     value<bool>(&viewport_fullscreen)->zero_tokens(),          "run in fullscreen mode");

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

#if 1
#include <fstream>
#include <boost/scoped_array.hpp>

bool convert_rmzn_file(const std::string& in_file_name,
                       const std::string& out_file_name,
                       const int          in_width,
                       const int          in_height,
                       const float        in_z_min,
                       const float        in_z_max,
                       const bool         in_invert)
{
    using namespace std;
    using boost::scoped_array;

    const int height = in_height;
    const int width  = in_width;

    fstream in_file;
    in_file.open(in_file_name.c_str(), ios_base::in);
    fstream out_file;
    out_file.open(out_file_name.c_str(), ios_base::out | ios_base::binary | ios_base::trunc);

    scoped_array<unsigned short>    out_buffer(new unsigned short[height * width]);

    if (!in_file) {
        cerr << "convert_rmzn_file(): error opening input file: " << in_file_name << std::endl;
        return (false);
    }

    if (!out_file) {
        cerr << "convert_rmzn_file(): error opening output file: " << out_file_name << std::endl;
        return (false);
    }

    std::size_t i = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float v;
            in_file >> v;

            if (v > 0.0f) {
                float vnrm = ((4000.0f - v) - in_z_min) / (in_z_max - in_z_min);
                //if (in_invert) vnrm = 1.0 - vnrm;
                v = vnrm * 65534.0f/*254.0f*/ + 1.0f;
            }
            else {
                v = 0.0f;
            }

            out_buffer[i++] = static_cast<unsigned short>(v);

            if ((y * width + x) % 1000 == 0) {
                cout << "done: " << std::fixed << static_cast<double>((y * width + x)) * 100.0 / (width * height) << "%";
                cout << "\xd";
            }
        }
    }

    in_file.close();
    
    out_file.write((char*)(out_buffer.get()), height * width * sizeof(unsigned short));
    if (out_file.bad()) {
        cerr << "convert_rmzn_file(): error writing output file" << std::endl;
        out_file.close();
        return (false);
    }

    out_file.close();

    return (true);
}
#endif

int main(int argc, char **argv)
{
    // some standard things to turn off
    std::ios_base::sync_with_stdio(false);

#if 0
    convert_rmzn_file("e:/data/height_field/horizons_mary_cole/raw/MJC_HorizonA1_521_1201MZN_grid_ASCIIMZN.rmzn",
                      "e:/data/height_field/horizons_mary_cole/raw/MJC_HorizonA1_volume_inverted.raw",
                      6861,
                      13737,
                      0.0f,//2788.18,
                      4000.0f,//3486.847,
                      true);
    convert_rmzn_file("e:/data/height_field/horizons_mary_cole/raw/MJC_HorizonB_1430_3668_grid_ASCIIMZN.rmzn",
                      "e:/data/height_field/horizons_mary_cole/raw/MJC_HorizonB1_volume_inverted.raw",
                      6863,
                      14297,
                      0.0f,//2788.18,
                      4000.0f,//3486.847,
                      true);
    convert_rmzn_file("e:/data/height_field/horizons_mary_cole/raw/MJC_HorizonC_1430_2653_grid_ASCIIMZN.rmzn",
                      "e:/data/height_field/horizons_mary_cole/raw/MJC_HorizonC1_volume_inverted.raw",
                      6875,
                      14301,
                      0.0f,//2788.18,
                      4000.0f,//3486.847,
                      true);
#endif

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#ifdef NDEBUG
    std::cout << "Release" << std::endl;
#else
    std::cout << "Debug" << std::endl;
#endif
    std::cout << "IDL: " << _ITERATOR_DEBUG_LEVEL << std::endl;
    std::cout << "SCL: " << _SECURE_SCL << std::endl;
    std::cout << "HID: " << _HAS_ITERATOR_DEBUGGING << std::endl;
#endif
    ///////////////////////////////////////////////////////////////////////////////////////////////
    using namespace scm;
    using namespace scm::gl;

    shared_ptr<core> scm_core(new core(argc, argv));
    QApplication     app(argc, argv);

    viewer::viewer_attributes   viewer_attribs;
    wm::surface::format_desc    window_format(FORMAT_RGBA_8,
                                              FORMAT_D24_S8,
                                              true /*double_buffer*/,
                                              false /*quad_buffer*/);
    wm::context::attribute_desc context_attribs(4, //SCM_GL_CORE_OPENGL_VERSION / 100,
                                                3, //SCM_GL_CORE_OPENGL_VERSION / 10 % 10,
                                                false /*compatibility*/,
                                                false /*debug*/,
                                                false /*forward*/);

    viewer_attribs._multi_samples = multi_samples;
    viewer_attribs._super_samples = 1;

    data::application_window* app_window(new data::application_window(height_field_file, viewport_size, viewer_attribs, context_attribs, window_format));

    app_window->setWindowTitle((std::string("schism: ") + scm_application_name).c_str());
    app_window->show();

    return app.exec();
}
