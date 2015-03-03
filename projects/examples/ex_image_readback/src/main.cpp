
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <fstream>
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
#include <scm/core/pointer_types.h>
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

static const std::string    scm_application_name = "schism example: image readback";

} // namespace

static bool initialize_cmd_line(scm::core& c)
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    options_description  cmd_options("program options");

    cmd_options.add_options()
        ("?",                                                                          "show this help message")
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


    std::string filename_to_nbr = "head4_w256_h256_d225_c1_b8";
    std::string file_ending = "png";

    unsigned x_res = 256;
    unsigned y_res = 256;
    unsigned z_slices = 224;

    unsigned max_dim = scm::math::max(scm::math::max(x_res, y_res), z_slices);

    float Ns = 96.078431;
    scm::math::vec3f Ka(0.0);
    scm::math::vec3f Kd(0.64);
    scm::math::vec3f Ks(0.5);
    float Ni = 1.0;
    float d = 0.0;
    int illum = 0;

    float z_step = ((float)z_slices / max_dim) / (float)z_slices;

    //newmtl head4_w256_h256_d225_c1_b80000
    //    Ns 96.078431
    //    Ka 0.000000 0.000000 0.000000
    //    Kd 0.640000 0.640000 0.640000
    //    Ks 0.500000 0.500000 0.500000
    //    Ni 1.000000
    //    d 0.000000
    //    illum 0
    //    map_Kd head4_w256_h256_d225_c1_b80000.png
    //    map_d head4_w256_h256_d225_c1_b80000.png

//# Blender v2.73 (sub 0) OBJ File : ''
//# www.blender.org
//    mtllib untitled.mtl
//        o head4_w256_h256_d225_c1_b80000_Plane.225
//        v - 0.768611 10.369658 0.925193
//        v 0.231389 10.369658 0.925193
//        v - 0.768611 10.369658 - 0.074807
//        v 0.231389 10.369658 - 0.074807
//        vt 0.000000 0.000000
//        vt 1.000000 0.000000
//        vt 1.000000 1.000000
//        vt 0.000000 1.000000
//        usemtl head4_w256_h256_d225_c1_b80000
//        s off
//        f 1 / 1 2 / 2 4 / 3 3 / 4

    std::ofstream volume_mtl;
    std::ofstream volume_obj;
    volume_obj.open("volume_to_object.obj");
    volume_mtl.open("volume_to_object.mtl");
    volume_mtl << "# Blender MTL File: 'None'\n # Material Count: " << z_slices << std::endl << std::endl;

    volume_obj << "# Blender v2.73 (sub 0) OBJ File : ''\n# www.blender.org\nmtllib volume_to_object.mtl" << std::endl;


    for (unsigned m = 0; m != z_slices; ++m){
        volume_mtl << "newmtl " << filename_to_nbr << m << "\n";

        volume_mtl << "Ns " << Ns << "\n";
        volume_mtl << "Ka " << Ka.x << " " << Ka.y << " " << Ka.z << "\n";
        volume_mtl << "Kd " << Kd.x << " " << Kd.y << " " << Kd.z << "\n";
        volume_mtl << "Ks " << Ks.x << " " << Ks.y << " " << Ks.z << "\n";
        volume_mtl << "Ni " << Ni << "\n";
        volume_mtl << "d " << d << "\n";
        volume_mtl << "illum " << illum << "\n";
        volume_mtl << "map_Kd " << filename_to_nbr;
        if (m < 10)
            volume_mtl << "000";
        else if (m < 100)
            volume_mtl << "00";
        else if (m < 1000)
            volume_mtl << "0";
        volume_mtl << m << "." << file_ending << "\n";

        volume_mtl << "map_d " << filename_to_nbr;
        if (m < 10)
            volume_mtl << "000";
        else if (m < 100)
            volume_mtl << "00";
        else if (m < 1000)
            volume_mtl << "0";
        volume_mtl << m << "." << file_ending << "\n";

        volume_mtl << std::endl;
        
        volume_obj << "o " << filename_to_nbr << m << "_Plane\n";
        volume_obj << "v " << 0.0 << " " << z_step * m << " " << (float)y_res / max_dim << std::endl;
        volume_obj << "v " << (float)y_res / max_dim << " " << z_step * m << " " << (float)y_res / max_dim << std::endl;
        volume_obj << "v " << 0.0 << " " << z_step * m << " " << 0.0 << std::endl;
        volume_obj << "v " << (float)y_res / max_dim << " " << z_step * m << " " << 0.0 << std::endl;
        volume_obj << "vt " << 0.0 << " " << 0.0 << std::endl;
        volume_obj << "vt " << 1.0 << " " << 0.0 << std::endl;
        volume_obj << "vt " << 1.0 << " " << 1.0 << std::endl;
        volume_obj << "vt " << 0.0 << " " << 1.0 << std::endl;
        volume_obj << "usemtl " << filename_to_nbr << m << std::endl;
        
    }


    volume_mtl.close();
    volume_obj.close();

    return 0;

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
    //QCleanlooksStyle*   st = new QCleanlooksStyle;
    //QPlastiqueStyle*    st = new QPlastiqueStyle;
    //QWindowsXPStyle*    st = new QWindowsXPStyle;
    //QWindowsVistaStyle* st = new QWindowsVistaStyle;

    //app.setStyle(st);

    viewer::viewer_attributes   viewer_attribs;
    wm::surface::format_desc    window_format(FORMAT_RGBA_8,
                                              FORMAT_D24_S8,
                                              true /*double_buffer*/,
                                              false /*quad_buffer*/);
    wm::context::attribute_desc context_attribs(SCM_GL_CORE_OPENGL_VERSION / 100,
                                                SCM_GL_CORE_OPENGL_VERSION / 10 % 10,
                                                false /*compatibility*/,
                                                false  /*debug*/,
                                                false /*forward*/);

    viewer_attribs._multi_samples = multi_samples;
    viewer_attribs._super_samples = 0;

    // data::application_window* app_window(new data::application_window(math::vec2ui(width, height), viewer_attribs, context_attribs, window_format));
    data::application_window* app_window(new data::application_window(viewport_size, viewer_attribs, context_attribs, window_format));

    app_window->setWindowTitle(scm_application_name.c_str());
    app_window->show();

    return app.exec();
}
