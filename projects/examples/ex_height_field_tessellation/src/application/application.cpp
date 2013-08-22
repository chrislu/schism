
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

#include <scm/log.h>
#include <scm/core/math.h>

#include <scm/gl_core.h>
#include <scm/gl_core/math.h>

#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/utilities/geometry_highlight.h>

#include <renderers/height_field_data.h>
#include <renderers/height_field_tessellator.h>

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
namespace data {

application_window::application_window(const std::string&                     input_file,
                                       const math::vec2ui&                    vp_size,
                                       const gl::viewer::viewer_attributes&   view_attrib,
                                       const gl::wm::context::attribute_desc& ctx_attrib,
                                       const gl::wm::surface::format_desc&    win_fmt)
  : gl::gui::viewer_window(vp_size, view_attrib, ctx_attrib, win_fmt)
  , _input_file(input_file)
  , _super_sample(false)
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
    shutdown();
    std::cout << "application_window::~application_window(): bye, bye..." << std::endl;
}

bool
application_window::init_renderer()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    _viewer->settings()._clear_color = vec4f(0.15f, 0.15f, 0.15f, 1.0f);

    // context
    const render_device_ptr& device = _viewer->device();

    device->main_context()->register_debug_callback(make_shared<scm_debug_output>());

    _hf_draw_wireframe = false;
    _hf_draw_quad_mesh = true;

    try {
        //_height_field = make_shared<height_field_data>(device,
        //                                               "./puget_sound_height_512_16bit.png",
        //                                               vec3f(1.0f, 1.0f, 0.04f));

        std::vector<std::string>    input_files;
        std::vector<vec3f>          bbox_min;
        std::vector<vec3f>          bbox_max;

        input_files.push_back(_input_file);
        //bbox_min.push_back(vec3f(3.0f,  779.0f, 1430.39f)); bbox_max.push_back(vec3f(6877.0f, 17485.0f, 2756.32f));
        bbox_min.push_back(vec3f::zero()); bbox_max.push_back(vec3f(1.0f, 1.0f, 0.1f));

        //input_files.push_back("e:/data_online/height_field/horizons_mcole/MJC_HorizonA_521_1201MZN_even.tif");
        //bbox_min.push_back(vec3f(3.0f, 3262.0f, 521.353f)); bbox_max.push_back(vec3f(6863.0f, 16998.0f, 1201.12f));

        //input_files.push_back("e:/data_online/height_field/horizons_mcole/MJC_HorizonB_1430_3668MZN_even.tif");
        //bbox_min.push_back(vec3f(3.0f, 2702.0f, 1429.56f)); bbox_max.push_back(vec3f(6865.0f, 16998.0f, 3667.64f));

        //input_files.push_back("e:/data_online/height_field/horizons_mcole/MJC_HorizonC_1430_2653MZN_even.tif");
        //bbox_min.push_back(vec3f(3.0f, 2700.0f, 1430.39f)); bbox_max.push_back(vec3f(6877.0f, 17000.0f, 2652.73f));

        vec3f scene_bbox_min  = vec3f( (std::numeric_limits<float>::max)());
        vec3f scene_bbox_max  = vec3f(-(std::numeric_limits<float>::max)());
        vec3f scene_bbox_size = vec3f(0.0f);

        for (size_t h = 0; h < input_files.size(); ++h) {
            scene_bbox_min.x = min(scene_bbox_min.x, bbox_min[h].x);
            scene_bbox_min.y = min(scene_bbox_min.y, bbox_min[h].y);
            scene_bbox_min.z = min(scene_bbox_min.z, bbox_min[h].z);

            scene_bbox_max.x = max(scene_bbox_max.x, bbox_max[h].x);
            scene_bbox_max.y = max(scene_bbox_max.y, bbox_max[h].y);
            scene_bbox_max.z = max(scene_bbox_max.z, bbox_max[h].z);
        }
        scene_bbox_size = scene_bbox_max - scene_bbox_min;
        float max_dim   = max(scene_bbox_size.x, max(scene_bbox_size.y, scene_bbox_size.z));

        for (size_t h = 0; h < input_files.size(); ++h) {
            vec3f bbox_size = (bbox_max[h] - bbox_min[h]) / max_dim;
            _height_fields.push_back(
                make_shared<height_field_data>(
                    device,
                    input_files[h].c_str(),
                    bbox_size));

            vec3f off = (bbox_min[h] - scene_bbox_min) / max_dim;
            _height_fields.back()->transform(make_scale(1.0f, 1.0f, -1.0f) * make_translation(off));
        }

        //_height_fields.push_back(make_shared<height_field_data>(device,
        //                                                        _input_file.c_str(),
        //                                                        vec3f(6874.0f, 16706.0f, 1325.93f) / 16706.0f));
        //_height_fields.push_back(make_shared<height_field_data>(device,
        //                                                        "e:/data_online/height_field/horizons_mcole/MJC_HorizonA_521_1201MZN_even.tif",
        //                                                        vec3f(6860.0f, 13736.0f, 688.767f) / 13736.0f));
        //_height_fields.push_back(make_shared<height_field_data>(device,
        //                                                        "e:/data_online/height_field/horizons_mcole/MJC_HorizonB_1430_3668MZN_even.tif",
        //                                                        vec3f(6862.0f, 14296.0f, 688.767f) / 14296.0f));
        //_height_fields.push_back(make_shared<height_field_data>(device,
        //                                                        "e:/data_online/height_field/horizons_mcole/MJC_HorizonC_1430_2653MZN_even.tif",
        //                                                        vec3f(6860.0f, 13736.0f, 688.767f) / 13736.0f));
        _height_field_renderer   = make_shared<height_field_tessellator>(device);
        _hf_mouse_over_highlight = make_shared<geometry_highlight>(device);
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "application_window::init_renderer(): unable to initialize the render system (" 
            << "evoking error: " << e.what() << ").";
        err() << msg.str() << log::end;
        return (false);
    }

    return (true);
}

void
application_window::shutdown()
{
    std::cout << "application_window::shutdown(): bye, bye..." << std::endl;

    _height_fields.clear();

    _height_field_renderer.reset();
    _hf_mouse_over_highlight.reset();
}

void
application_window::update(const gl::render_device_ptr& device,
                           const gl::render_context_ptr& context)
{
    _height_field_renderer->update_main_camera(context, _viewer->main_camera());
}

void
application_window::display(const gl::render_context_ptr& context)
{
    using namespace scm::gl;
    using namespace scm::math;

    const mat4f& view_matrix         = _viewer->main_camera().view_matrix();
    const mat4f& proj_matrix         = _viewer->main_camera().projection_matrix();

    { // multi sample pass
        //context_state_objects_guard csg(context);
        //context_program_guard       cpg(context);

        height_field_tessellator::draw_mode  hf_draw_mode = _hf_draw_wireframe ? height_field_tessellator::MODE_WIRE_FRAME : height_field_tessellator::MODE_SOLID;
        height_field_tessellator::mesh_mode  hf_mesh_mode = _hf_draw_quad_mesh ? height_field_tessellator::MODE_QUAD_PATCHES : height_field_tessellator::MODE_TRIANGLE_PATCHES;

        for (size_t h = 0; h < _height_fields.size(); ++h) {
            _height_field_renderer->draw(context, _height_fields[h], _super_sample, hf_mesh_mode, hf_draw_mode);

            mat4f mv_matrix = view_matrix * _height_fields[h]->transform();
            _hf_mouse_over_highlight->draw(context, _height_fields[h]->bbox_geometry(),
                                           proj_matrix, mv_matrix,
                                           geometry::MODE_WIRE_FRAME, vec4f(0.0f, 0.4f, 1.0f, 1.0f), 2.0f);
        }

        if (_hf_mouse_over) {
            mat4f mv_matrix = view_matrix * _hf_mouse_over->transform();
            _hf_mouse_over_highlight->draw(context, _hf_mouse_over->bbox_geometry(),
                                           proj_matrix, mv_matrix,
                                           geometry::MODE_WIRE_FRAME, vec4f(0.0f, 1.0f, 0.3f, 1.0f), 2.0f);
        }

    }
}

void
application_window::reshape(const gl::render_device_ptr& device,
                            const gl::render_context_ptr& context,
                            int w, int h)
{
    using namespace scm::gl;
    using namespace scm::math;

    viewer_window::reshape(device, context, w, h);
}


void
application_window::keyboard_input(int k, bool state, scm::uint32 mod)
{
    viewer_window::keyboard_input(k, state, mod);

    static const float pixel_tolerance_fact = 2.0f;
    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_Escape:    close_program();break;
            case Qt::Key_S:         _super_sample = !_super_sample;break;
            case Qt::Key_A:
                _height_field_renderer->pixel_tolerance(_height_field_renderer->pixel_tolerance() / pixel_tolerance_fact);
                out() << "pixel_tolerance: " << std::fixed << std::setprecision(2) << _height_field_renderer->pixel_tolerance() << log::end;
                break;
            case Qt::Key_Z:
                _height_field_renderer->pixel_tolerance(_height_field_renderer->pixel_tolerance() * pixel_tolerance_fact);
                out() << "pixel_tolerance: " << std::fixed << std::setprecision(2) << _height_field_renderer->pixel_tolerance() << log::end;
                break;
            case Qt::Key_W:         _hf_draw_wireframe = !_hf_draw_wireframe; break;
            case Qt::Key_Q:         _hf_draw_quad_mesh = !_hf_draw_quad_mesh; break;
            case Qt::Key_K:
                {
                    std::fstream    m_out("tmp_view.out", std::ios_base::out);
                    m_out << _viewer->main_camera().view_matrix();
                    m_out.close();
                }
                break;
            case Qt::Key_L:
                {
                    std::fstream    m_in("tmp_view.out", std::ios_base::in);
                    math::mat4f     view_matrix;
                    m_in >> view_matrix;
                    _viewer->main_camera().view_matrix(view_matrix);
                    m_in.close();
                }
                break;
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
    //height_field_data_ptr prev_mouse_over = _hf_mouse_over;
    _hf_mouse_over = pick_box_instance(x, y);
}

const height_field_data_ptr
application_window::pick_box_instance(int x, int y) const
{
    math::vec3f tmp_hit;
    return (pick_box_instance(x, y, tmp_hit));
}

const height_field_data_ptr
application_window::pick_box_instance(int x, int y, math::vec3f& out_hit) const
{
    using namespace scm::gl;
    using namespace scm::math;

    vec2f   nrm_coords = _viewer->norm_viewport_coords(vec2i(x, y));
    ray     pick_ray   = _viewer->main_camera().generate_ray(nrm_coords);

    vec3f   nearest_hit;
    vec3f   nearest_hit_obj;
    float   nearest_hit_dist = (std::numeric_limits<float>::max)();
    
    height_field_data_ptr picked_inst;

    // temporary until we actually have more boxes
    std::list<height_field_data_ptr> _boxes;
    std::for_each(_height_fields.begin(), _height_fields.end(),
        [&_boxes](const height_field_data_ptr& p) -> void {
            _boxes.push_back(p);
    });

    std::list<height_field_data_ptr>::const_iterator i = _boxes.begin();
    std::list<height_field_data_ptr>::const_iterator e = _boxes.end();

    for (; i != e; ++i) {
        const height_field_data_ptr& cur_inst = *i;
        vec3f hit_entry;
        vec3f hit_exit;
        const mat4f& cur_xform = cur_inst->transform();

        ray obj_pick_ray(pick_ray);
        obj_pick_ray.transform_preinverted(cur_xform);

        if (cur_inst->bbox().intersect(obj_pick_ray, hit_entry, hit_exit)) {
            vec3f hit = vec3f(cur_xform * vec4f(hit_entry, 1.0));
            float dst = length_sqr(hit - pick_ray.origin());
            if (dst < nearest_hit_dist) {
                nearest_hit      = hit;
                nearest_hit_obj  = hit_entry;
                nearest_hit_dist = dst;
                picked_inst      = cur_inst;
            }
        }
    }

    if (picked_inst) {
        out_hit = nearest_hit_obj;
    }

    return (picked_inst);
}

} // namespace data
} // namespace scm
