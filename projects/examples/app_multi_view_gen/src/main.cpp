
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/assign/list_of.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <scm/core.h>
#include <scm/log.h>
#include <scm/core/pointer_types.h>
#include <scm/core/io/tools.h>
#include <scm/core/time/accumulate_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_core.h>

#include <scm/gl_util/imaging/texture_loader.h>
#include <scm/gl_util/manipulators/trackball_manipulator.h>
#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/primitives/wavefront_obj.h>

//#include <GL/gl.h>
#include <GL/freeglut.h>
//#include <GL/gl3.h>

static bool fraps_bug = true;

static int winx = 1600;
static int winy = 1024;

const scm::math::vec3f diffuse(0.7f, 0.7f, 0.7f);
const scm::math::vec3f specular(0.2f, 0.7f, 0.9f);
const scm::math::vec3f ambient(0.1f, 0.1f, 0.1f);
const scm::math::vec3f position(1, 1, 1);

class demo_app
{
public:
    demo_app() {
        _initx = 0;
        _inity = 0;

        _lb_down = false;
        _mb_down = false;
        _rb_down = false;

        _dolly_sens = 10.0f;

        _projection_matrix = scm::math::mat4f::identity();
    }
    virtual ~demo_app();

    bool initialize();
    void display();
    void resize(int w, int h);
    void mousefunc(int button, int state, int x, int y);
    void mousemotion(int x, int y);
    void keyboard(unsigned char key, int x, int y);

private:
    scm::gl::trackball_manipulator _trackball_manip;
    float _initx;
    float _inity;

    bool _lb_down;
    bool _mb_down;
    bool _rb_down;

    float _dolly_sens;


    scm::shared_ptr<scm::gl::render_device>     _device;
    scm::shared_ptr<scm::gl::render_context>    _context;

    scm::gl::program_ptr        _shader_program;

    scm::gl::buffer_ptr         _index_buffer;
    scm::gl::vertex_array_ptr   _vertex_array;

    scm::math::mat4f            _projection_matrix;

    scm::shared_ptr<scm::gl::box_geometry>  _box;
    scm::shared_ptr<scm::gl::wavefront_obj_geometry>  _obj;
    scm::gl::depth_stencil_state_ptr     _dstate_less;
    scm::gl::depth_stencil_state_ptr     _dstate_disable;

    scm::gl::blend_state_ptr            _no_blend;
    scm::gl::blend_state_ptr            _blend_omsa;
    scm::gl::blend_state_ptr            _color_mask_green;

    scm::gl::texture_2d_ptr             _color_texture;

    scm::gl::sampler_state_ptr          _filter_lin_mip;
    scm::gl::sampler_state_ptr          _filter_aniso;
    scm::gl::sampler_state_ptr          _filter_nearest;
    scm::gl::sampler_state_ptr          _filter_linear;

    scm::gl::texture_2d_ptr             _color_buffer;
    scm::gl::texture_2d_ptr             _color_buffer_resolved;
    scm::gl::texture_2d_ptr             _depth_buffer;
    scm::gl::frame_buffer_ptr           _framebuffer;
    scm::gl::frame_buffer_ptr           _framebuffer_resolved;
    scm::shared_ptr<scm::gl::quad_geometry>  _quad;
    scm::gl::program_ptr                _pass_through_shader;
    scm::gl::depth_stencil_state_ptr    _depth_no_z;
    scm::gl::rasterizer_state_ptr       _ms_back_cull;

    scm::gl::timer_query_ptr            _timer_view_gen;
    scm::gl::timer_query_ptr            _timer_draw;

}; // class demo_app


namespace  {

scm::scoped_ptr<demo_app> _application;

} // namespace


demo_app::~demo_app()
{
    _shader_program.reset();
    _index_buffer.reset();
    _vertex_array.reset();

    _box.reset();
    _obj.reset();

    _filter_lin_mip.reset();
    _filter_aniso.reset();
    _filter_nearest.reset();
    _color_texture.reset();

    _filter_linear.reset();
    _color_buffer.reset();
    _depth_buffer.reset();
    _framebuffer.reset();
    _quad.reset();
    _pass_through_shader.reset();
    _depth_no_z.reset();
    _ms_back_cull.reset();
    _color_buffer_resolved.reset();
    _framebuffer_resolved.reset();

    _timer_view_gen.reset();
    _timer_draw.reset();

    _context.reset();
    _device.reset();
}

bool
demo_app::initialize()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;



    _device.reset(new scm::gl::render_device());
    _context = _device->main_context();




    _shader_program = _device->create_program(
        list_of(_device->create_shader_from_file(STAGE_VERTEX_SHADER,   "../../../src/shaders/phong_lighting.glslv"))
               (_device->create_shader_from_file(STAGE_GEOMETRY_SHADER, "../../../src/shaders/phong_lighting.glslg"))
               (_device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/shaders/phong_lighting.glslf")));

    if (!_shader_program) {
        scm::err() << "error creating shader program" << log::end;
        return (false);
    }

    _shader_program->uniform("light_ambient", ambient);
    _shader_program->uniform("light_diffuse", diffuse);
    _shader_program->uniform("light_specular", specular);
    _shader_program->uniform("light_position", position);

    _shader_program->uniform("material_ambient", ambient);
    _shader_program->uniform("material_diffuse", diffuse);
    _shader_program->uniform("material_specular", specular);
    _shader_program->uniform("material_shininess", 128.0f);
    _shader_program->uniform("material_opacity", 1.0f);





    scm::out() << *_device << scm::log::end;

#if 0
    std::vector<scm::math::vec3f>   positions;
    std::vector<scm::math::vec3f>   normals;
    std::vector<unsigned short>              indices;

    positions.push_back(scm::math::vec3f(0.0f, 0.0f, 0.0f));
    positions.push_back(scm::math::vec3f(1.0f, 0.0f, 0.0f));
    positions.push_back(scm::math::vec3f(1.0f, 1.0f, 0.0f));
    positions.push_back(scm::math::vec3f(0.0f, 1.0f, 0.0f));

    normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));
    normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));
    normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));
    normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));

    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);
    indices.push_back(0);
    indices.push_back(2);
    indices.push_back(3);

    buffer_ptr positions_buf;
    buffer_ptr normals_buf;

    positions_buf = _device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, positions.size() * sizeof(scm::math::vec3f), &positions.front());
    normals_buf   = _device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, normals.size() * sizeof(scm::math::vec3f), &normals.front());
    _index_buffer = _device->create_buffer(BIND_INDEX_BUFFER,  USAGE_STATIC_DRAW, indices.size() * sizeof(unsigned short), &indices.front());

    _vertex_array = _device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F)(1, 1, TYPE_VEC3F),
                                                 list_of(positions_buf)(normals_buf));
#else
    std::vector<scm::math::vec3f>   positions_normals;
    std::vector<unsigned short>              indices;

    positions_normals.push_back(scm::math::vec3f(0.0f, 0.0f, 0.0f));
    positions_normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));

    positions_normals.push_back(scm::math::vec3f(1.0f, 0.0f, 0.0f));
    positions_normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));

    positions_normals.push_back(scm::math::vec3f(1.0f, 1.0f, 0.0f));
    positions_normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));

    positions_normals.push_back(scm::math::vec3f(0.0f, 1.0f, 0.0f));
    positions_normals.push_back(scm::math::vec3f(0.0f, 0.0f, 1.0f));

    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);
    indices.push_back(0);
    indices.push_back(2);
    indices.push_back(3);

    buffer_ptr positions_normals_buf;

    positions_normals_buf = _device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, positions_normals.size() * sizeof(scm::math::vec3f), &positions_normals.front());
    _index_buffer = _device->create_buffer(BIND_INDEX_BUFFER,  USAGE_STATIC_DRAW, indices.size() * sizeof(unsigned short), &indices.front());

    _vertex_array = _device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F, 2 * sizeof(scm::math::vec3f))
                                                              (0, 1, TYPE_VEC3F, 2 * sizeof(scm::math::vec3f)),
                                                 list_of(positions_normals_buf));
#endif

    _dstate_less    = _device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    depth_stencil_state_desc dstate = _dstate_less->descriptor();
    dstate._depth_test = false;

    _dstate_disable = _device->create_depth_stencil_state(dstate);
    //_dstate_disable = _device->create_depth_stencil_state(false);
    _no_blend           = _device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _blend_omsa         = _device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);
    _color_mask_green   = _device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO,
                                                             EQ_FUNC_ADD, EQ_FUNC_ADD, COLOR_GREEN | COLOR_BLUE);

    _box.reset(new box_geometry(_device, vec3f(-0.5f), vec3f(0.5f)));
    //_obj.reset(new wavefront_obj_geometry(_device, "e:/working_copies/external_libraries/gpucast/app/gl3pp_uniformbuffer/mini.obj"));
    _obj.reset(new wavefront_obj_geometry(_device, "e:/devel/schism/resources/objects/guardian.obj"));
    //_obj.reset(new wavefront_obj_geometry(_device, "e:/working_copies/ncc1701d/ncc1701d.obj"));
    //_obj.reset(new wavefront_obj_geometry(_device, "g:/data/geometry/EM HDModels Car/clio/OBJ/horizon.obj"));


    texture_loader tex_loader;
    //_color_texture = tex_loader.load_texture_2d(*_device, "e:/working_copies/schism_x64/resources/textures/16bit_rgb_test.png" , true);
    _color_texture = tex_loader.load_texture_2d(*_device,
        "e:/working_copies/schism_x64/resources/textures/DH213SN.hdr", true, true);
        //"e:/working_copies/schism_x64/resources/textures/angel_512.jpg", true, true);

    _filter_lin_mip = _device->create_sampler_state(FILTER_MIN_MAG_MIP_LINEAR, WRAP_CLAMP_TO_EDGE);
    _filter_aniso   = _device->create_sampler_state(FILTER_ANISOTROPIC, WRAP_CLAMP_TO_EDGE, 16);
    _filter_nearest = _device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _filter_linear  = _device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);

    // initialize framebuffer /////////////////////////////////////////////////////////////////////
    _color_buffer          = _device->create_texture_2d(vec2ui(winx, winy) * 1, FORMAT_RGBA_8, 1, 1, 1);
    _depth_buffer          = _device->create_texture_2d(vec2ui(winx, winy) * 1, FORMAT_D24, 1, 1, 1);
    _framebuffer           = _device->create_frame_buffer();
    _framebuffer->attach_color_buffer(0, _color_buffer);
    _framebuffer->attach_depth_stencil_buffer(_depth_buffer);

    _color_buffer_resolved = _device->create_texture_2d(vec2ui(winx, winy) * 1, FORMAT_RGBA_8);
    _framebuffer_resolved  = _device->create_frame_buffer();
    _framebuffer_resolved->attach_color_buffer(0, _color_buffer_resolved);

    _quad.reset(new quad_geometry(_device, vec2f(0.0f, 0.0f), vec2f(1.0f, 1.0f)));
    _depth_no_z   = _device->create_depth_stencil_state(false, false);
    _ms_back_cull = _device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    std::string v_pass = "\
        #version 330\n\
        \
        uniform mat4 mvp;\
        out vec2 tex_coord;\
        layout(location = 0) in vec3 in_position;\
        layout(location = 2) in vec2 in_texture_coord;\
        void main()\
        {\
            gl_Position = mvp * vec4(in_position, 1.0);\
            tex_coord = in_texture_coord;\
        }\
        ";
    std::string f_pass = "\
        #version 330\n\
        \
        in vec2 tex_coord;\
        uniform sampler2D in_texture;\
        layout(location = 0) out vec4 out_color;\
        void main()\
        {\
        out_color = texelFetch(in_texture, ivec2(gl_FragCoord.xy), 0).rgba;\n\
        }\
        ";
    ////texture(in_texture, tex_coord);
    _pass_through_shader = _device->create_program(list_of(_device->create_shader(STAGE_VERTEX_SHADER, v_pass))
                                                          (_device->create_shader(STAGE_FRAGMENT_SHADER, f_pass)));

    _timer_view_gen = _device->create_timer_query();
    _timer_draw     = _device->create_timer_query();

    _trackball_manip.dolly(2.5f);

    return (true);
}

unsigned plah = 0;
double  time_view_gen = 0.0;
double  time_draw     = 0.0;
int     time_accum_count = 0;

void
demo_app::display()
{
    using namespace scm::gl;
    using namespace scm::math;

    // clear the color and depth buffer
    //glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    mat4f    view_matrix         = _trackball_manip.transform_matrix();
    mat4f    model_matrix        = make_scale(0.01f, 0.01f, 0.01f);//mat4f::identity();
    mat4f    model_view_matrix   = view_matrix * model_matrix;
    mat4f    mv_inv_transpose    = transpose(inverse(model_view_matrix));

    _shader_program->uniform("projection_matrix", _projection_matrix);
    _shader_program->uniform("model_view_matrix", model_view_matrix);
    _shader_program->uniform("model_view_matrix_inverse_transpose", mv_inv_transpose);
    _shader_program->uniform("color_texture_aniso", 0);
    _shader_program->uniform("color_texture_nearest", 1);

    _context->clear_default_color_buffer(FRAMEBUFFER_BACK, vec4f(.2f, .2f, .2f, 1.0f));
    _context->clear_default_depth_stencil_buffer();

    _context->reset();

    _context->begin_query(_timer_view_gen);
    { // multi sample pass
        context_state_objects_guard csg(_context);
        context_texture_units_guard tug(_context);
        context_framebuffer_guard   fbg(_context);

        _context->clear_color_buffer(_framebuffer, 0, vec4f( .2f, .2f, .2f, 1.0f));
        _context->clear_depth_stencil_buffer(_framebuffer);

        _context->set_frame_buffer(_framebuffer);
        //_context->set_viewport(viewport(vec2ui(0, 0), 1 * vec2ui(winx, winy)));
        // build array of two viewports (lower left and upper right)
        vec2f o = vec2f(0.0f, 0.0f);
        vec2f s = vec2f(static_cast<float>(winx), static_cast<float>(winy));
        vec2f m = s * 0.5f;
        // viewport has an () operator which returns a viewport array,
        // so it is easy to concatenate viewports this way viewport(vp_1)(vp_2)(vp_3)...(vp_n)
        // you also can directly use viewport_array like this viewport_array(vp_1)(vp_2)(vp_3)...(vp_n)
        //_context->set_viewports(viewport(o, m)(m, m));
        _context->set_viewport(viewport(o, s));

        _context->set_depth_stencil_state(_dstate_less);
        _context->set_blend_state(_no_blend);
        _context->set_rasterizer_state(_ms_back_cull);

        _context->bind_program(_shader_program);

        //_context->bind_texture(_color_texture, _filter_aniso,   0);
        //_context->bind_texture(_color_texture, _filter_nearest, 1);

        //_box->draw(_context, geometry::MODE_SOLID);
        _obj->draw(_context);
    }
    _context->end_query(_timer_view_gen);

    //_context->resolve_multi_sample_buffer(_framebuffer, _framebuffer_resolved);
    //_context->generate_mipmaps(_color_buffer_resolved);

    _context->reset();

    _context->begin_query(_timer_draw);
    { // fullscreen pass
        mat4f   pass_mvp = mat4f::identity();
        ortho_matrix(pass_mvp, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

        _pass_through_shader->uniform("in_texture", 0);
        _pass_through_shader->uniform("mvp", pass_mvp);

        _context->set_default_frame_buffer();

        _context->set_depth_stencil_state(_depth_no_z);
        _context->set_blend_state(_no_blend);

        _context->bind_program(_pass_through_shader);

        //_context->bind_texture(_color_buffer_resolved, _filter_nearest, 0);
        _context->bind_texture(_color_buffer, _filter_nearest, 0);

        _quad->draw(_context);
    }
    _context->end_query(_timer_draw);

    _context->collect_query_results(_timer_view_gen);
    _context->collect_query_results(_timer_draw);

    time_view_gen    += scm::time::to_milliseconds(scm::time::nanosec(_timer_view_gen->result()));
    time_draw        += scm::time::to_milliseconds(scm::time::nanosec(_timer_draw->result()));
    time_accum_count += 1;

    if ((time_view_gen + time_draw) > 1000.0) {
        boost::io::ios_all_saver ias(std::cout);
        std::cout << "view_gen time:  " << std::fixed << time_view_gen / time_accum_count  << "msec "
                  << "(fps: " << static_cast<double>(time_accum_count * 1000) / time_view_gen << "), "
                  << "view_draw time: " << std::fixed << time_draw / time_accum_count << "msec"
                  << "(fps: " << static_cast<double>(time_accum_count * 1000) / time_draw << "), "
                  << "combined time: " << std::fixed <<  (time_accum_count + time_draw) / time_accum_count  << "msec "
                  << "(fps: " << static_cast<double>(time_accum_count * 1000) / (time_accum_count + time_draw) << ")" << std::endl;
        time_view_gen    = 0.0;
        time_draw        = 0.0;
        time_accum_count = 0;
    }

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();

    if (fraps_bug) {
        if (glGetError() != GL_NO_ERROR) {
            std::cout << "fraps bug after swap handled" << std::endl;
        }
        else {
            //std::cout << "nothing after swap" << std::endl;
        }
        fraps_bug = false;
    }
}

void
demo_app::resize(int w, int h)
{
    // safe the new dimensions
    winx = w;
    winy = h;

    // set the new viewport into which now will be rendered
    using namespace scm::gl;
    using namespace scm::math;

    _context->set_viewport(viewport(vec2ui(0, 0), vec2ui(w, h)));

    scm::math::perspective_matrix(_projection_matrix, 60.f, float(w)/float(h), 0.1f, 100.0f);
}

void
demo_app::mousefunc(int button, int state, int x, int y)
{
    switch (button) {
        case GLUT_LEFT_BUTTON:
            {
                _lb_down = (state == GLUT_DOWN) ? true : false;
            }break;
        case GLUT_MIDDLE_BUTTON:
            {
                _mb_down = (state == GLUT_DOWN) ? true : false;
            }break;
        case GLUT_RIGHT_BUTTON:
            {
                _rb_down = (state == GLUT_DOWN) ? true : false;
            }break;
    }

    _initx = 2.f * float(x - (winx/2))/float(winx);
    _inity = 2.f * float(winy - y - (winy/2))/float(winy);
}

void
demo_app::mousemotion(int x, int y)
{
    float nx = 2.f * float(x - (winx/2))/float(winx);
    float ny = 2.f * float(winy - y - (winy/2))/float(winy);

    //std::cout << "nx " << nx << " ny " << ny << std::endl;

    if (_lb_down) {
        _trackball_manip.rotation(_initx, _inity, nx, ny);
    }
    if (_rb_down) {
        _trackball_manip.dolly(_dolly_sens * (ny - _inity));
    }
    if (_mb_down) {
        _trackball_manip.translation(nx - _initx, ny - _inity);
    }

    _inity = ny;
    _initx = nx;
}

void
demo_app::keyboard(unsigned char key, int x, int y)
{
}

void
glut_display()
{
    if (_application)
        _application->display();

}

void
glut_resize(int w, int h)
{
    if (_application)
        _application->resize(w, h);
}

void
glut_mousefunc(int button, int state, int x, int y)
{
    if (_application)
        _application->mousefunc(button, state, x, y);
}

void
glut_mousemotion(int x, int y)
{
    if (_application)
        _application->mousemotion(x, y);
}

void
glut_idle()
{
    glutPostRedisplay();
}

void
glut_keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        // ESC key
        case 27: {
            _application.reset();
            std::cout << "reset application" << std::endl;
            exit (0);
                 }
            break;
        case 'f':
            glutFullScreenToggle();
            break;
        default:
            _application->keyboard(key, x, y);
    }
}


int main(int argc, char **argv)
{
    std::ios_base::sync_with_stdio(false);

    scm::shared_ptr<scm::core>      scm_core(new scm::core(argc, argv));
    _application.reset(new demo_app());

    // the stuff that has to be done
    glutInit(&argc, argv);
    glutInitContextVersion(4, 2);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    //glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);

    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_MULTISAMPLE);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("simple_glut");

    // init the GL context
    if (!_application->initialize()) {
        std::cout << "error initializing gl context" << std::endl;
        return (-1);
    }

    // set the callbacks for resize, draw and idle actions
    glutReshapeFunc(glut_resize);
    glutDisplayFunc(glut_display);
    glutKeyboardFunc(glut_keyboard);
    glutMouseFunc(glut_mousefunc);
    glutMotionFunc(glut_mousemotion);
    glutIdleFunc(glut_idle);

    // and finally start the event loop
    glutMainLoop();

    return (0);
}
