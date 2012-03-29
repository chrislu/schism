
#include "application.h"

#include <QtGui/QMenu>
#include <QtGui/QMenuBar>

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <QtGui/QFileDialog>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <scm/core/platform/windows.h>
#include <cuda_gl_interop.h>

#include <scm/log.h>
#include <scm/core/math.h>

#include <scm/gl_core.h>
#include <scm/gl_core/math.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>
#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/box_volume.h>
#include <scm/gl_util/utilities/coordinate_cross.h>
#include <scm/gl_util/utilities/geometry_highlight.h>

#include <application/volume_gui/volume_data_dialog.h>

#include <renderer/volume_data.h>
#include <renderer/volume_renderer.h>
#include <renderer/cuda_volume_data.h>
#include <renderer/cuda_volume_renderer.h>

namespace {

const scm::math::vec3f diffuse(0.7f, 0.7f, 0.7f);
const scm::math::vec3f specular(0.2f, 0.7f, 0.9f);
const scm::math::vec3f ambient(0.1f, 0.1f, 0.1f);
const scm::math::vec3f position(1, 1, 1);

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
namespace gl {
namespace gui {
} // namespace gui
} // namespace gl

namespace data {

application_window::application_window(const math::vec2ui&                    vp_size,
                                       const gl::viewer::viewer_attributes&   view_attrib,
                                       const gl::wm::context::attribute_desc& ctx_attrib,
                                       const gl::wm::surface::format_desc&    win_fmt)
  : gl::gui::viewer_window(vp_size, view_attrib, ctx_attrib, win_fmt)
  , _viewport_size(vp_size)
  , _show_raw(false)
  , _use_opencl_renderer(false)
  , _volume_data_dialog(0)
{
    if (!init_renderer()) {
        std::stringstream msg;
        msg << "application_window::application_window(): error initializing multi large image rendering stystem.";
        err() << msg.str() << log::end;
        throw (std::runtime_error(msg.str()));
    }

    // file menu
    QMenu*          vis_menu       = new QMenu("Visualization", this);
    QAction*        load_vol       = vis_menu->addAction("open volume...");
                                     vis_menu->addSeparator();
    QAction*        vol_prop       = vis_menu->addAction("volume data properties...");

    connect(load_vol,  SIGNAL(triggered()), this, SLOT(open_volume()));
    connect(vol_prop,  SIGNAL(triggered()), this, SLOT(open_volume_data_dialog()));
    _main_menubar->addMenu(vis_menu);
}

application_window::~application_window()
{
    shutdown();
    std::cout << "application_window::~application_window(): bye, bye..." << std::endl;
}


bool
application_window::init_renderer()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    _viewer->settings()._clear_color      = vec4f(0.0f, 0.0f, 0.0f, 1.0f);


    if (!_viewer->device()->enable_cuda_interop()) {
        err() << log::error
              << "application_window::init_renderer(): unable to initialize CUDA system."
              << log::end;
        return false;
    }

    const render_device_ptr& device = _viewer->device();

    device->main_context()->register_debug_callback(make_shared<scm_debug_output>());

    volume_data::color_map_type cmap;
    volume_data::alpha_map_type amap;

#if 1
    amap.add_stop(0,       1.0f);
    amap.add_stop(0.42f,   0.5f);
    amap.add_stop(0.48f,   0.0f);
    amap.add_stop(0.52f,   0.0f);
    amap.add_stop(0.58f,   0.5f);
    amap.add_stop(1.0f,    1.0f);
#elif 1
    amap.add_stop(0,       1.0f);
    amap.add_stop(0.33f,   0.5f);
    amap.add_stop(0.40f,   0.0f);
    amap.add_stop(0.60f,   0.0f);
    amap.add_stop(0.66f,   0.5f);
    amap.add_stop(1.0f,    1.0f);
#else
    amap.add_stop(0,       0.0f);
    amap.add_stop(1.0f,    1.0f);
#endif

#if 0
    // blue-grey-orange
    cmap.add_stop(0,       vec3f(0.0f, 1.0f, 1.0f));
    cmap.add_stop(0.25f,   vec3f(0.0f, 0.0f, 1.0f));
    cmap.add_stop(0.375f,  vec3f(0.256637f, 0.243243f, 0.245614f));
    cmap.add_stop(0.50f,   vec3f(0.765487f, 0.738739f, 0.72807f));
    cmap.add_stop(0.625f,  vec3f(0.530973f, 0.27027f, 0.0f));
    cmap.add_stop(0.75f,   vec3f(1.0f, 0.333333f, 0.0f));
    cmap.add_stop(1.0f,    vec3f(1.0f, 1.0f, 0.0f));
#else
    // blue-white-red
    cmap.add_stop(0.0f, vec3f(0.0f, 0.0f, 1.0f));
    cmap.add_stop(0.5f, vec3f(1.0f, 1.0f, 1.0f));
    cmap.add_stop(1.0f, vec3f(1.0f, 0.0f, 0.0f));
#endif

    //std::string vfile = "/home/chrislu/devel/data/wfarm_200_w512_h439_d512_c1_b8.raw";
    //std::string vfile = "e:/data/volume/vrgeo/parihaka/pari_full_rm_8float_bri.sgy";
    //std::string vfile = "Z:/volume_data/vrgeo/parihaka_new_zealand/source_data/volume/pari_full_rm_8float_bri.vol";
    //std::string vfile = "e:/data/volume/vrgeo/new_zealand/volumes/pari_full_rm_8float_bri_TRIMMED.vol";
    //std::string vfile = "g:/volume/vrgeo/eni_fp_volume/test_data3far.segy";
    //std::string vfile = "e:/data/volume/vrgeo/reflect.vol";
    //std::string vfile = "e:/data/volume/vrgeo/gfaks.vol";
    //std::string vfile = "e:/data/volume/vrgeo/wfarm_200_w512_h439_d512_c1_b8.raw";

    try {
        _coord_cross      = make_shared<coordinate_cross>(device, 0.15f);
        _volume_highlight = make_shared<geometry_highlight>(device);

        //_volume_data.reset(new volume_data(device, vfile, cmap, amap));
        //_volume_data_cuda.reset(new cuda_volume_data(device, _volume_data));

        _volume_renderer.reset(new volume_renderer(device));
        _volume_renderer_cuda.reset(new cuda_volume_renderer(device, _viewport_size));
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "application_window::init_renderer(): unable to initialize the render system ("
            << "evoking error: " << e.what() << ").";
        err() << msg.str() << log::end;
        return false;
    }


    // text rendering
    try {
        font_face_ptr output_font(new font_face(device, "../../../res/fonts/Segoeui.ttf", 18, 1.5f, font_face::smooth_lcd));
        _text_renderer  = make_shared<text_renderer>(device);
        _output_text    = make_shared<text>(device, output_font, font_face::style_bold, "sick, sad world...");

        mat4f   fs_projection = make_ortho_matrix(0.0f, static_cast<float>(_viewport_size.x),
                                                  0.0f, static_cast<float>(_viewport_size.y), -1.0f, 1.0f);
        _text_renderer->projection_matrix(fs_projection);

        _output_text->text_color(math::vec4f(1.0f, 1.0f, 0.0f, 1.0f));
        _output_text->text_outline_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));
        _output_text->text_kerning(true);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(std::string("vtexture_system::vtexture_system(): ") + e.what());
    }

    return true;
}

void
application_window::shutdown()
{
    if (_volume_data_dialog) {
        _volume_data_dialog->hide();
        delete _volume_data_dialog;
    }
    _coord_cross.reset();
    _volume_highlight.reset();

    _volume_renderer.reset();
    _volume_data_cuda.reset();
    _volume_data.reset();

    _output_text.reset();
    _text_renderer.reset();
}

void
application_window::update(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context)
{
    _viewer->main_camera().projection_perspective(60.f, float(_viewport_size.x) / float(_viewport_size.y), 0.01f, 10.0f);

    _volume_renderer->update(context, _viewer->main_camera());

    if (_volume_data) {
        _volume_data->update(context, _viewer->main_camera());
        _volume_data_cuda->update(context);
    }
}

void
application_window::display(const gl::render_context_ptr& context)
{
    using namespace scm::gl;
    using namespace scm::math;

    const mat4f& view_matrix         = _viewer->main_camera().view_matrix();
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    { // draw volumes
        context_framebuffer_guard cfg(context);

        //_coord_cross->draw(context, proj_matrix, view_matrix, 4.0f);

        if (_volume_data) {
            vec2f o = vec2f(0.0f, 0.0f);
            vec2f s = vec2f(_viewport_size.x, _viewport_size.y);

            context->set_viewport(viewport(o, s));

#if 0
            mat4f mv_matrix = view_matrix * _volume_data->transform();
            _volume_highlight->draw(context, _volume_data->bbox_geometry(),
                                            proj_matrix, mv_matrix,
                                            geometry::MODE_WIRE_FRAME, vec4f(0.0f, 1.0f, 0.3f, 1.0f), 2.0f);
#endif
            if (_use_opencl_renderer) {
                _volume_renderer_cuda->draw(context, _volume_data_cuda);
                _volume_renderer_cuda->present(context);
            }
            else {
                _volume_renderer->draw(context, _volume_data, _show_raw ? volume_renderer::volume_raw : volume_renderer::volume_color_map);
            }
        }
    }

    if (_volume_data) { // text overlay
        vec3ui lod_size = util::mip_level_dimensions(_volume_data->data_dimensions(), static_cast<unsigned>(_volume_data->selected_lod()));
        {
            std::stringstream   os;
            os << std::fixed << std::setprecision(3)
               << (_use_opencl_renderer ? "CUDA " : "OpenGL ")
               << (_show_raw ? "Raw Volume " : "Color-Mapped Volume ")
               << "LOD: " << _volume_data->selected_lod()
               << ", size: " << lod_size << std::endl;
            _output_text->text_string(os.str());
            //_text_renderer->draw_shadowed(context, vec2i(10, 10), _output_text);
            _text_renderer->draw_outlined(context, vec2i(10, 10), _output_text);
        }
    }
}

void
application_window::reshape(const gl::render_device_ptr& device,
                            const gl::render_context_ptr& context,
                            int w, int h)
{
    viewer_window::reshape(device, context, w, h);
}

void
application_window::keyboard_input(int k, bool state, scm::uint32 mod)
{
    viewer_window::keyboard_input(k, state, mod);

    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_Escape:    close_program();break;
            case Qt::Key_Space:     _show_raw = !_show_raw;break;
            case Qt::Key_C:         _use_opencl_renderer = !_use_opencl_renderer; break;
            case Qt::Key_S:         _volume_renderer->reload_shaders(_viewer->device());
                                    //_volume_renderer_cuda->reload_kernels(_viewer->device());
            case Qt::Key_1:         _volume_data->selected_lod(_volume_data->selected_lod() - 0.25f);break;
            case Qt::Key_2:         _volume_data->selected_lod(_volume_data->selected_lod() + 0.25f);break;
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
    //out() << "application_window::mouse_double_click()" << log::end;
}

void
application_window::mouse_press(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_press()" << log::end;
}

void
application_window::mouse_release(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_release()" << log::end;
}

void
application_window::mouse_move(gl::viewer::mouse_button b, int x, int y)
{
    //out() << "application_window::mouse_move()" << log::end;
}

void
application_window::open_volume_data_dialog()
{
    if (_volume_data) {
        if (_volume_data_dialog == 0) {
            _volume_data_dialog = new scm::gui::volume_data_dialog(_volume_data, this);
            _volume_data_dialog->setWindowTitle("volume data properties");
        }

        _volume_data_dialog->show();
    }
}

void
application_window::open_volume()
{
    out() << "application_window::open_volume()" << log::end;

    std::string image_file = QFileDialog::getOpenFileName(this,
                                                          "Open volume...",
                                                          0,
                                                          "All (*.raw *.vol *.sgy *.segy);; .raw (*.raw);; .vol (*.vol);; .segy (*.sgy *.segy)").toStdString();

    if (!image_file.empty()) {

        using namespace scm::math;

        volume_data::color_map_type cmap;
        volume_data::alpha_map_type amap;

#if 1
        amap.add_stop(0,       1.0f);
        amap.add_stop(0.42f,   0.5f);
        amap.add_stop(0.48f,   0.0f);
        amap.add_stop(0.52f,   0.0f);
        amap.add_stop(0.58f,   0.5f);
        amap.add_stop(1.0f,    1.0f);
#elif 1
        amap.add_stop(0,       1.0f);
        amap.add_stop(0.33f,   0.5f);
        amap.add_stop(0.40f,   0.0f);
        amap.add_stop(0.60f,   0.0f);
        amap.add_stop(0.66f,   0.5f);
        amap.add_stop(1.0f,    1.0f);
#else
        amap.add_stop(0,       0.0f);
        amap.add_stop(1.0f,    1.0f);
#endif

#if 0
        // blue-grey-orange
        cmap.add_stop(0,       vec3f(0.0f, 1.0f, 1.0f));
        cmap.add_stop(0.25f,   vec3f(0.0f, 0.0f, 1.0f));
        cmap.add_stop(0.375f,  vec3f(0.256637f, 0.243243f, 0.245614f));
        cmap.add_stop(0.50f,   vec3f(0.765487f, 0.738739f, 0.72807f));
        cmap.add_stop(0.625f,  vec3f(0.530973f, 0.27027f, 0.0f));
        cmap.add_stop(0.75f,   vec3f(1.0f, 0.333333f, 0.0f));
        cmap.add_stop(1.0f,    vec3f(1.0f, 1.0f, 0.0f));
#else
        // blue-white-red
        cmap.add_stop(0.0f, vec3f(0.0f, 0.0f, 1.0f));
        cmap.add_stop(0.5f, vec3f(1.0f, 1.0f, 1.0f));
        cmap.add_stop(1.0f, vec3f(1.0f, 0.0f, 0.0f));
#endif

        try {
            _volume_data.reset(new volume_data(_viewer->device(), image_file, cmap, amap));
            _volume_data_cuda.reset(new cuda_volume_data(_viewer->device(), _volume_data));
        }
        catch (std::exception& e) {
            err() << "application_window::open_volume(): error opening file: ('" << image_file << "') "
                  << "error: " << e.what() << log::end;
            _volume_data.reset();
            _volume_data_cuda.reset();
        }

        if (_volume_data_dialog != 0) {
            _volume_data_dialog->volume(_volume_data);
        }
    }
}

} // namespace data
} // namespace scm
