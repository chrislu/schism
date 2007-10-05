
#include <iostream>

#include <string>

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include <scm/core.h>
#include <scm/ogl.h>
#include <scm/ogl/gl.h>
#include <GL/glut.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>
#include <scm/ogl/time/time_query.h>
#include <scm/core/utilities/foreach.h>

#include <image_handling/image_loader.h>

#include <deferred_shading/deferred_shader.h>
#include <deferred_shading/geometry.h>

boost::scoped_ptr<scm::deferred_shader>     _deferred_shader;

scm::gl::trackball_manipulator _trackball_manip;

int winx = 1024;
int winy = 640;
bool fullscreen = false;

float initx = 0;
float inity = 0;

bool lb_down = false;
bool mb_down = false;
bool rb_down = false;

float dolly_sens = 10.0f;

bool display_gbuffers = false;
unsigned display_buffer = 0;

// texture objects ids
unsigned tex0_id = 0;
unsigned tex1_id = 0;

const double    timer_screen_update = 500.0; // msec

static const unsigned light_num = 4;
bool lights_state[light_num] = {true, false, false, false};


void define_lights()
{
    float one[4]    = {1.0f, 1.0f, 1.0f, 1.0f};
    float zro[4]    = {0.0f, 0.0f, 0.0f, 1.0f};

    float dif_sun[4]    = {0.2, 0.2, 0.2, 1};
    float spc_sun[4]    = {0.2, 0.2, 0.2, 1};
    float amb_sun[4]    = {0.01, 0.01, 0.01, 1};
    float att_sun[4]    = {1.0, 0.0, 0.0, 1};//{1.0, 0.25, 0.125, 1};
    float spot_sun[2]   = {64, 180};

    float dif_torch[4]    = {0.25, 0.25, 0.2, 1};
    float spc_torch[4]    = {0.3, 0.3, 0.2, 1};
    float amb_torch[4]    = {0.005, 0.005, 0.005, 1};
    float att_torch[4]    = {1.0, 0.0, 0.0, 1};//{0.9, 0.125, 0.025, 1};
    float spot_torch[2]   = {96, 60};

    math::mat_glf_t perf_to_gl = math::mat4f_identity;

    perf_to_gl.rotate(-90.0f, 1.0f, 0.0f, 0.0f);

    math::vec4f_t pos_sun1      = perf_to_gl * math::vec4f_t(-1.7, -5.0, 4.0, 0);
    math::vec4f_t pos_sun2      = perf_to_gl * math::vec4f_t(1.7,  -5.0, 4.0, 0);

    math::vec4f_t pos_torch1    = perf_to_gl * math::vec4f_t(3.0, -3.0, 1.0, 1.0);//1.6, -0.1, 2.6, 1.0);
    math::vec4f_t pos_torch2    = perf_to_gl * math::vec4f_t(-3.0, -3.0, 1.0, 1.0);//-1.5, 0.25, 2.5, 1.0);

    math::vec4f_t dir_sun1      = perf_to_gl * math::vec4f_t(1,-1,-1,0);
    math::vec4f_t dir_sun2      = perf_to_gl * math::vec4f_t(-1,-1,-1,0);

    math::vec4f_t dir_torch1    = perf_to_gl * math::vec4f_t(-3.0, 3.0, -1.0, 1.0);//-1, 0.7, -0.7, 0);
    math::vec4f_t dir_torch2    = perf_to_gl * math::vec4f_t(3.0,  3.0, -1.0, 1.0);//1, -0.1, -0.4, 0);


    // setup light 0 - sun 1
    glLightfv(GL_LIGHT0, GL_SPECULAR, spc_sun);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  amb_sun);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  dif_sun);
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  att_sun[0]);
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION,    att_sun[1]);
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, att_sun[2]);
    //glLightf(GL_LIGHT0, GL_SPOT_EXPONENT , spot_sun[0]);
    //glLightf(GL_LIGHT0, GL_SPOT_CUTOFF,    spot_sun[1]);
    //glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION , dir_sun1.vec_array);
    glLightfv(GL_LIGHT0, GL_POSITION, pos_sun1.vec_array);


    // setup light 0 - sun 2
    glLightfv(GL_LIGHT1, GL_SPECULAR, spc_sun);
    glLightfv(GL_LIGHT1, GL_AMBIENT,  amb_sun);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  dif_sun);
    glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION,  att_sun[0]);
    glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION,    att_sun[1]);
    glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, att_sun[2]);
    //glLightf(GL_LIGHT1, GL_SPOT_EXPONENT , spot_sun[0]);
    //glLightf(GL_LIGHT1, GL_SPOT_CUTOFF,    spot_sun[1]);
    //glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION , dir_sun2.vec_array);
    glLightfv(GL_LIGHT1, GL_POSITION, pos_sun2.vec_array);

    // setup light 0 - torch 1
    glLightfv(GL_LIGHT2, GL_SPECULAR, spc_torch);
    glLightfv(GL_LIGHT2, GL_AMBIENT,  amb_torch);
    glLightfv(GL_LIGHT2, GL_DIFFUSE,  dif_torch);
    glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION,  att_torch[0]);
    glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION,    att_torch[1]);
    glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, att_torch[2]);
    glLightf(GL_LIGHT2, GL_SPOT_EXPONENT , spot_torch[0]);
    glLightf(GL_LIGHT2, GL_SPOT_CUTOFF,    spot_torch[1]);
    glLightfv(GL_LIGHT2, GL_SPOT_DIRECTION , dir_torch1.vec_array);
    glLightfv(GL_LIGHT2, GL_POSITION, pos_torch1.vec_array);

    // setup light 0 - torch 2
    glLightfv(GL_LIGHT3, GL_SPECULAR, spc_torch);
    glLightfv(GL_LIGHT3, GL_AMBIENT,  amb_torch);
    glLightfv(GL_LIGHT3, GL_DIFFUSE,  dif_torch);
    glLightf(GL_LIGHT3, GL_CONSTANT_ATTENUATION,  att_torch[0]);
    glLightf(GL_LIGHT3, GL_LINEAR_ATTENUATION,    att_torch[1]);
    glLightf(GL_LIGHT3, GL_QUADRATIC_ATTENUATION, att_torch[2]);
    glLightf(GL_LIGHT3, GL_SPOT_EXPONENT , spot_torch[0]);
    glLightf(GL_LIGHT3, GL_SPOT_CUTOFF,    spot_torch[1]);
    glLightfv(GL_LIGHT3, GL_SPOT_DIRECTION , dir_torch2.vec_array);
    glLightfv(GL_LIGHT3, GL_POSITION, pos_torch2.vec_array);

    for (unsigned l = 0; l < light_num; ++l) {
        if (lights_state[l]) {
            glEnable(GL_LIGHT0 + l);
        }
        else {
            glDisable(GL_LIGHT0 + l);
        }
    }

}

bool init_gl()
{
    // check for opengl verison 2.0 with
    // opengl shading language support
    if (!scm::ogl.get().is_supported("GL_VERSION_2_0")) {
        std::cout << "GL_VERSION_2_0 not supported" << std::endl;
        return (false);
    }
    if (!scm::ogl.get().is_supported("GL_EXT_framebuffer_object")) {
        std::cout << "GL_EXT_framebuffer_object not supported" << std::endl;
        return (false);
    }
    if (!scm::ogl.get().is_supported("GL_ARB_draw_buffers")) {
        std::cout << "GL_ARB_draw_buffers not supported" << std::endl;
        return (false);
    }
    if (!scm::ogl.get().is_supported("GL_ARB_texture_float")) {
        std::cout << "GL_ARB_texture_float not supported" << std::endl;
        return (false);
    }

    std::cout << (char*)glGetString(GL_VERSION) << std::endl;

//    int max_samples = 0;
//    glGetIntegerv(GL_MAX_SAMPLES_EXT, &max_samples);
//    std::cout << "max. samples for multisample fbo: " << max_samples << std::endl;

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    //glClearColor(0.2f,0.2f,0.2f,1);
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(4.5);

    //image ps_data;

    //// load texture image
    //if (!open_image("./../../../res/textures/rock_decal_gloss_512.tga", ps_data)) {
    //    return (false);
    //}
    //// generate texture object id
    //glGenTextures(1, &tex0_id);
    //if (tex0_id == 0) {
    //    std::cout << "error creating texture object" << std::endl;
    //    return (false);
    //}
    //// load image as texture to texture object
    //if (!load_2d_texture(tex0_id, ps_data, false)) {
    //    std::cout << "error uploading texture" << std::endl;
    //    return (false);
    //}
    //// delete image
    //close_image(ps_data);

    //// load texture image
    //if (!open_image("./../../../res/textures/rock_normal_512.tga", ps_data)) {
    //    std::cout << "error open_image " << std::endl;
    //    return (false);
    //}
    //// generate texture object id
    //glGenTextures(1, &tex1_id);
    //if (tex1_id == 0) {
    //    std::cout << "error creating texture object" << std::endl;
    //    return (false);
    //}
    //// load image as texture to texture object
    //if (!load_2d_texture(tex1_id, ps_data, false)) {
    //    std::cout << "error uploading texture" << std::endl;
    //    return (false);
    //}
    //// delete image
    //close_image(ps_data);

    //// setup light 0
    //glLightfv(GL_LIGHT0, GL_SPECULAR, one);//spc);
    //glLightfv(GL_LIGHT0, GL_AMBIENT,  amb);
    //glLightfv(GL_LIGHT0, GL_DIFFUSE,  one);//dif);
    //glLightfv(GL_LIGHT0, GL_POSITION, pos);

    //glEnable(GL_LIGHT0);

    //// define material parameters
    //glMaterialfv(GL_FRONT, GL_SPECULAR, spc);
    //glMaterialf(GL_FRONT, GL_SHININESS, 128.0f);
    //glMaterialfv(GL_FRONT, GL_AMBIENT, amb);
    //glMaterialfv(GL_FRONT, GL_DIFFUSE, dif);


    _deferred_shader.reset(new scm::deferred_shader(winx, winy));
#if 1
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/carriage.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/door_left.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/door_right.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/hood.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/interior.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/trunk.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/mfrd/Data/beetle/wheel_all.sgeom")) {
        return (false);
    }

    unsigned fc = 0;
    foreach(const geometry& geom, _geometries) {
        fc += geom._face_count;
    }

    std::cout << "overall face count: " << fc << std::endl;
#endif
    return (true);
}

void shutdown_gl()
{
    glDeleteTextures(1, &tex0_id);
    glDeleteTextures(1, &tex1_id);
}

void draw_stuff()
{
    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        _trackball_manip.apply_transform();

        define_lights();

        math::mat_glf_t perf_to_gl = math::mat4f_identity;

        perf_to_gl.rotate(-90.0f, 1.0f, 0.0f, 0.0f);

        glTranslatef(0,-1,0);
        glRotatef(7, 1, 0, 0);
        glRotatef( 7, 0, 0, 1);
        glRotatef(45, 0, 1, 0);
        glTranslatef(-.5,0,-.5);
        glMultMatrixf(perf_to_gl.mat_array);

#if 0
        // activate texture unit 0
        glActiveTexture(GL_TEXTURE0);
        // enable texturing on unit 0
        glEnable(GL_TEXTURE_2D);
        // bind texture to unit 0
        glBindTexture(GL_TEXTURE_2D, tex0_id);

        // activate texture unit 1
        glActiveTexture(GL_TEXTURE1);
        // enable texturing on unit 1
        glEnable(GL_TEXTURE_2D);
        // bind texture to unit 1
        glBindTexture(GL_TEXTURE_2D, tex1_id);

        // draw primitive
        glPushMatrix();
            glTranslatef(-0.5, -0.5, 0);
            // draw quad in immediate mode
            // with texture coordinates associated with every vertex
            glBegin(GL_QUADS);
                glMultiTexCoord2f(GL_TEXTURE0, 0.0f, 0.0f);
                glVertex2f(0.0f, 0.0f);

                glMultiTexCoord2f(GL_TEXTURE0, 1.0f, 0.0f);
                glVertex2f(1.0f, 0.0f);

                glMultiTexCoord2f(GL_TEXTURE0, 1.0f, 1.0f);
                glVertex2f(1.0f, 1.0f);

                glMultiTexCoord2f(GL_TEXTURE0, 0.0f, 1.0f);
                glVertex2f(0.0f, 1.0f);
            glEnd();
        glPopMatrix();

        // unbind all texture objects and disable texturing on texture units
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
#else
    foreach(const geometry& geom, _geometries) {

        glPushMatrix();
        glTranslatef(geom._desc._geometry_origin.x,
                     geom._desc._geometry_origin.y,
                     geom._desc._geometry_origin.z);

        geom._vbo->bind();

        for (unsigned db = 0; db < geom._indices.size(); ++db) {
            geom._indices[db]->bind();

            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   geom._materials[db]._Ka.vec_array);
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   geom._materials[db]._Kd.vec_array);
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  geom._materials[db]._Ks.vec_array);
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS,  geom._materials[db]._Ns > 128.0f ? 128.0f : geom._materials[db]._Ns);

            geom._indices[db]->draw_elements();
            geom._indices[db]->unbind();
        }

        geom._vbo->unbind();
        glPopMatrix();
    }
#endif
    // restore previously saved modelview matrix
    glPopMatrix();

}

void display()
{
    static scm::gl::time_query          _gl_timer_fill;
    static scm::gl::time_query          _gl_timer_light;
    static double                       _gl_accum_time_fill     = 0.0;
    static double                       _gl_accum_time_light    = 0.0;
    static double                       _accum_time             = 0.0;
    static unsigned                     _accum_count            = 0;


    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    _gl_timer_fill.start(); {

        _deferred_shader->start_fill_pass(); {
            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
            draw_stuff();
        }
        _deferred_shader->end_fill_pass();
    }
    _gl_timer_fill.stop();


    _gl_timer_light.start(); {

        if (display_gbuffers) {
            _deferred_shader->display_buffers();
        }
        else {

            if (display_buffer == 0) {
                _deferred_shader->shade();
            }
            else {
                _deferred_shader->display_buffer(display_buffer - 1);
            }
        }
    }
    _gl_timer_light.stop();

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();

    _gl_timer_fill.collect_result();
    _gl_timer_light.collect_result();

    _gl_accum_time_fill     += scm::time::to_milliseconds(_gl_timer_fill.get_time());
    _gl_accum_time_light    += scm::time::to_milliseconds(_gl_timer_light.get_time());

    _accum_time             += (scm::time::to_milliseconds(_gl_timer_fill.get_time())
                              + scm::time::to_milliseconds(_gl_timer_light.get_time()));
    ++_accum_count;

    if (_accum_time > timer_screen_update) {
        std::stringstream   output;

        output.precision(2);
        output << std::fixed << "fill_pass: " << _gl_accum_time_fill / static_cast<double>(_accum_count) << "msec "
                             << "light_pass: " << _gl_accum_time_light / static_cast<double>(_accum_count) << "msec "
                             << std::endl;

        std::cout << output.str();

        _accum_time             = 0.0;
        _gl_accum_time_fill     = 0.0;
        _gl_accum_time_light    = 0.0;
        _accum_count            = 0;
    }
}

void resize(int w, int h)
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    // safe the new dimensions
    winx = w;
    winy = h;

    // set the new viewport into which now will be rendered
    glViewport(0, 0, w, h);
    
    // reset the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.f, float(w)/float(h), 1.5f, 10.f);

    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void mousefunc(int button, int state, int x, int y)
{
    switch (button) {
        case GLUT_LEFT_BUTTON:
            {
                lb_down = (state == GLUT_DOWN) ? true : false;
            }break;
        case GLUT_MIDDLE_BUTTON:
            {
                mb_down = (state == GLUT_DOWN) ? true : false;
            }break;
        case GLUT_RIGHT_BUTTON:
            {
                rb_down = (state == GLUT_DOWN) ? true : false;
            }break;
    }

    initx = 2.f * float(x - (winx/2))/float(winx);
    inity = 2.f * float(winy - y - (winy/2))/float(winy);
}

void mousemotion(int x, int y)
{
    float nx = 2.f * float(x - (winx/2))/float(winx);
    float ny = 2.f * float(winy - y - (winy/2))/float(winy);

    //std::cout << "nx " << nx << " ny " << ny << std::endl;

    if (lb_down) {
        _trackball_manip.rotation(initx, inity, nx, ny);
    }
    if (rb_down) {
        _trackball_manip.dolly(dolly_sens * (ny - inity));
    }
    if (mb_down) {
        _trackball_manip.translation(nx - initx, ny - inity);
    }

    inity = ny;
    initx = nx;
}

void idle()
{
    // on ilde just trigger a redraw
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 'd':
        case 'D': display_gbuffers = !display_gbuffers;break;
        case 'o':
        case 'O': open_geometry();break;
        case '1': lights_state[0] = !lights_state[0];break;
        case '2': lights_state[1] = !lights_state[1];break;
        case '3': lights_state[2] = !lights_state[2];break;
        case '4': lights_state[3] = !lights_state[3];break;
        case 'q': display_buffer = 0;break;
        case 'w': display_buffer = 1;break;
        case 'e': display_buffer = 2;break;
        case 'r': display_buffer = 3;break;
        case 't': display_buffer = 4;break;
        // ESC key
        case 27: exit (0); break;
        default:;
    }
}


int main(int argc, char **argv)
{
    int width;
    int height;
    bool fs;

    try {
        boost::program_options::options_description  cmd_options("program options");

        cmd_options.add_options()
            ("help", "show this help message")
            ("width", boost::program_options::value<int>(&width)->default_value(1024), "output width")
            ("height", boost::program_options::value<int>(&height)->default_value(640), "output height")
            ("fullscreen", boost::program_options::value<bool>(&fs)->default_value(false), "run in fullscreen mode");

        boost::program_options::variables_map       command_line;
        boost::program_options::parsed_options      parsed_cmd_line =  boost::program_options::parse_command_line(argc, argv, cmd_options);
        boost::program_options::store(parsed_cmd_line, command_line);
        boost::program_options::notify(command_line);

        if (command_line.count("help")) {
            std::cout << "usage: " << std::endl;
            std::cout << cmd_options;
            return (0);
        }
        winx = width;
        winy = height;
        fullscreen = fs;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return (-1);
    }
    // the stuff that has to be done
    glutInit(&argc, argv);
    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_STENCIL);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("OpenGL - simple deferred shading");

    if (fullscreen) {
        glutFullScreen();
    }
    // init the GL context
    if (!scm::initialize()) {
        std::cout << "error initializing scm library" << std::endl;
        return (-1);
    }

    // init the GL context
    if (!scm::gl::initialize()) {
        std::cout << "error initializing gl library" << std::endl;
        return (-1);
    }
    if (!init_image_loader()) {
        std::cout << "error initializing image library" << std::endl;
        return (-1);
    }
    if (!init_gl()) {
        std::cout << "error initializing gl context" << std::endl;
        return (-1);
    }

    
    // set the callbacks for resize, draw and idle actions
    glutReshapeFunc(resize);
    glutMouseFunc(mousefunc);
    glutMotionFunc(mousemotion);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutIdleFunc(idle);

    // and finally start the event loop
    glutMainLoop();

    shutdown_gl();

    return (0);
}



