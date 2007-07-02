

#include <iostream>
#include <map>

#include <boost/scoped_array.hpp>
#include <boost/assign/std/vector.hpp>

#include <scm/core.h>
#include <scm/ogl.h>
#include <scm/ogl/gl.h>
#include <scm/ogl/utilities/error_checker.h>
#include <GL/glut.h>

#include <scm/ogl/shader_objects/program_object.h>
#include <scm/ogl/shader_objects/shader_object.h>
#include <scm/ogl/vertexbuffer_object/vertexbuffer_object.h>

#include <scm/data/geometry/wavefront_obj/obj_file.h>
#include <scm/data/geometry/wavefront_obj/obj_loader.h>
#include <scm/data/geometry/wavefront_obj/obj_to_vertex_array.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>

#include <scm/input/tracking/art_dtrack.h>
#include <scm/input/tracking/target.h>

#include <scm/core/math/math.h>

scm::gl::trackball_manipulator _trackball_manip;
scm::gl::vertexbuffer_object   _obj_vbo;

int winx = 1024;
int winy = 640;

float initx = 0;
float inity = 0;

bool lb_down = false;
bool mb_down = false;
bool rb_down = false;

float dolly_sens = 10.0f;
float anim_angl = 0.0f;

struct vertex_format
{
    float       tex[3];
    float       pos[3];
};

namespace scm {

struct screen
{
    math::vec2f_t       _screen_world_dim;              // in meters
    math::vec3f_t       _screen_lower_left_location;    // in world coordinates
}; // struct screen

math::mat4f_t           _tracking_to_world_transform;
math::mat4f_t           _world_to_screen_transform;

screen                  _screen;

float                   _znear = 0.5f;  // 50cm
float                   _zfar  = 10.0f; // 10m

} // namespace scm

boost::scoped_ptr<scm::gl::program_object>   _shader_program;
boost::scoped_ptr<scm::gl::shader_object>    _vertex_shader;
boost::scoped_ptr<scm::gl::shader_object>    _fragment_shader;


boost::scoped_array<vertex_format>  vertices;
boost::scoped_array<unsigned short> indices;

boost::scoped_ptr<scm::inp::art_dtrack> _dtrack;

scm::inp::tracker::target_container     _targets;

// vbo ids
unsigned cube_vbo           = 0;
unsigned cube_index_buffer  = 0;

scm::data::wavefront_model obj_f;


bool init_gl()
{
    // check for opengl verison 1.5 with
    // vertex buffer objects support
    if (!scm::ogl.get().is_supported("GL_VERSION_1_5")) {
        std::cout << "GL_VERSION_1_5 not supported" << std::endl;
        return (false);
    }

    // set clear color, which is used to fill the background on glClear
    glClearColor(0.2f,0.2f,0.2f,1);
    
    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(1);

    _shader_program.reset(new scm::gl::program_object());
    _vertex_shader.reset(new scm::gl::shader_object(GL_VERTEX_SHADER));
    _fragment_shader.reset(new scm::gl::shader_object(GL_FRAGMENT_SHADER));

    // load shader code from files
    if (!_vertex_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/two_sided_vert.glsl")) {
        std::cout << "Error loadong vertex shader:" << std::endl;
        return (false);
    }
    if (!_fragment_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/two_sided_frag.glsl")) {
        std::cout << "Error loadong frament shader:" << std::endl;
        return (false);
    }

    // compile shaders
    if (!_vertex_shader->compile()) {
        std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
        std::cout << _vertex_shader->get_compiler_output() << std::endl;
        return (false);
    }
    if (!_fragment_shader->compile()) {
        std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
        std::cout << _fragment_shader->get_compiler_output() << std::endl;
        return (false);
    }

    // attatch shaders to program object
    if (!_shader_program->attach_shader(*_vertex_shader)) {
        std::cout << "unable to attach vertex shader to program object:" << std::endl;
        return (false);
    }
    if (!_shader_program->attach_shader(*_fragment_shader)) {
        std::cout << "unable to attach fragment shader to program object:" << std::endl;
        return (false);
    }

    // link program object
    if (!_shader_program->link()) {
        std::cout << "Error linking program - linker output:" << std::endl;
        std::cout << _shader_program->get_linker_output() << std::endl;
       return (false);
    }

    std::size_t obj_no = 0;
    std::size_t obj_to = 0;
    std::size_t obj_va_s = 0;
    std::size_t obj_ia_s = 0;

    if (!scm::data::open_obj_file("./../../../res/objects/some_objects.obj", obj_f)) {
        std::cout << "failed to parse obj" << std::endl;
    }
    else {
        std::cout << "done parsing obj" << std::endl;

    }

    //system("pause");

    boost::shared_array<float>          obj_va;
    boost::shared_array<scm::core::uint32_t> obj_ia;

    scm::data::generate_vertex_buffer(obj_f,
                                      obj_va,
                                      obj_va_s,
                                      obj_no,
                                      obj_to,
                                      obj_ia,
                                      obj_ia_s);

    std::cout << "num faces: " << (obj_ia_s / 3) << std::endl;


    using namespace scm::gl;
    using namespace boost::assign;


    vertex_layout::element_container vert_elem;

    vert_elem += vertex_element(vertex_element::position,
                                vertex_element::dt_vec3f);
    vert_elem += vertex_element(vertex_element::normal,
                                vertex_element::dt_vec3f);

    vertex_layout      vert_lo(vert_elem);
    element_layout     elem_lo(element_layout::triangles,
                               element_layout::dt_uint);

    if (!_obj_vbo.vertex_data(obj_va_s,
                              vert_lo,
                              obj_va.get())) {
        std::cout << "error uploading vertex data" << std::endl;
        return (false);
    }

     if (!_obj_vbo.element_data(obj_ia_s,
                                elem_lo,
                                obj_ia.get())) {
        std::cout << "error uploading element data" << std::endl;
        return (false);
    }

    obj_va.reset();
    obj_ia.reset();

    float dif[4]    = {0.7, 0.7, 0.7, 1};
    float spc[4]    = {0.2, 0.7, 0.9, 1};
    float amb[4]    = {0.1, 0.1, 0.1, 1};
    float pos[4]    = {1,1,1,0};

    float difm[4]    = {0.7, 0.7, 0.7, 1};
    float spcm[4]    = {0.2, 0.7, 0.9, 1};
    float ambm[4]    = {0.1, 0.1, 0.1, 1};


    glLightfv(GL_LIGHT0, GL_SPECULAR, spc);
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
    glLightfv(GL_LIGHT0, GL_POSITION, pos);

    // define material parameters
    glMaterialfv(GL_FRONT, GL_SPECULAR, spcm);
    glMaterialf(GL_FRONT, GL_SHININESS, 128.0f);
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambm);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, difm);


    _dtrack.reset(new scm::inp::art_dtrack());

    std::cout << "initializing art_dtrack" << std::endl;
    if (!_dtrack->initialize()) {
        return (false);
    }
    else {
        std::cout << " - successfullty initialized art_dtrack" << std::endl;
    }

    _targets.insert(scm::inp::tracker::target_container::value_type(6, scm::inp::target(6)));

    scm::_screen._screen_world_dim              = math::vec2f_t(3.0f, 2.0f);
    scm::_screen._screen_lower_left_location    = math::vec3f_t(-scm::_screen._screen_world_dim.x/2.0f,
                                                                 2.4f,
                                                                 0.45f);

    scm::_tracking_to_world_transform           = math::mat4f_identity;
    scm::_tracking_to_world_transform.m12       = 0.0f;     // x off
    scm::_tracking_to_world_transform.m13       = 0.0f;     // y off
    scm::_tracking_to_world_transform.m14       = -0.72f;    // z off 72cm

    scm::_world_to_screen_transform             = math::mat4f_identity;
    scm::_world_to_screen_transform.m12         = scm::_screen._screen_lower_left_location.x + scm::_screen._screen_world_dim.x / 2.0f;
    scm::_world_to_screen_transform.m13         = scm::_screen._screen_lower_left_location.y;
    scm::_world_to_screen_transform.m14         = scm::_screen._screen_lower_left_location.z + scm::_screen._screen_world_dim.y / 2.0f;


    return (true);
}

void draw_wavefront_obj_vertex_buffer_object()
{
    // retrieve current polygon mode
    GLint current_poly_mode[2];
    glGetIntegerv(GL_POLYGON_MODE, current_poly_mode);
    // set polygon mode to wireframe rendering
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    glColor3f(0.2f, 0.5f, 0.2f);
    _shader_program->bind();

    _obj_vbo.bind();
    glPushMatrix();
        _obj_vbo.draw_elements();
    glPopMatrix();
    _obj_vbo.unbind();

    _shader_program->unbind();
    // restore saved polygon mode
    glPolygonMode(GL_FRONT, current_poly_mode[0]);
    glPolygonMode(GL_BACK,  current_poly_mode[1]);

}

void display()
{
    // clear the color and depth buffer
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    // set shade model
    glShadeModel(GL_SMOOTH);
    //glShadeModel(GL_FLAT);

    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        //_trackball_manip.apply_transform();

        glTranslatef(0, 0, -10);

        //draw_wavefront_obj_vertex_buffer_object();

    // restore previously saved modelview matrix
    glPopMatrix();
    glLineWidth(5.0f);
    // push current modelview 
    glPushMatrix();
        glTranslatef(0, 1, 0);
        //glScalef(3,3,3);
            glBegin(GL_LINES);
            glColor3f( 1, 0, 0);
            glVertex3f(0, 0, 0);
            glVertex3f(1, 0, 0);

            glColor3f( 0, 1, 0);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 1, 0);

            glColor3f( 0, 0, 1);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0, 1);
            glEnd();
    // restore previously saved modelview matrix
    glPopMatrix();


    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();
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
    gluPerspective(60.f, float(w)/float(h), 0.1f, 100.f);


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

void shutdown_gl()
{
    _dtrack->shutdown();

    scm::gl::shutdown();
    scm::shutdown();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        // ESC key
        case 27: shutdown_gl();exit (0); break;
        default:;
    }
}

void idle()
{
    _dtrack->update(_targets);

    scm::inp::tracker::target_container::const_iterator target_it = _targets.find(6);

    if (target_it != _targets.end()) {
        //std::cout << target_it->second.transform().m00 << " " << target_it->second.transform().m04 << " " << target_it->second.transform().m08 << " " << target_it->second.transform().m12 << std::endl;
        //std::cout << target_it->second.transform().m01 << " " << target_it->second.transform().m05 << " " << target_it->second.transform().m09 << " " << target_it->second.transform().m13 << std::endl;
        //std::cout << target_it->second.transform().m02 << " " << target_it->second.transform().m06 << " " << target_it->second.transform().m10 << " " << target_it->second.transform().m14 << std::endl;
        //std::cout << target_it->second.transform().m03 << " " << target_it->second.transform().m07 << " " << target_it->second.transform().m11 << " " << target_it->second.transform().m15 << std::endl;

        math::vec4f_t   view_pos(target_it->second.transform().m12 / 1000.0f,
                                 target_it->second.transform().m13 / 1000.0f,
                                 target_it->second.transform().m14 / 1000.0f,
                                 1.0);

        view_pos = math::inverse(scm::_tracking_to_world_transform) * view_pos;

        // retrieve current matrix mode
        GLint  current_matrix_mode;
        glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

        // reset the projection matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        //float d = view_pos.y;

        view_pos = math::inverse(scm::_world_to_screen_transform) * view_pos;
        float d = math::abs(view_pos.y);

        glFrustum((view_pos.x - scm::_screen._screen_world_dim.x / 2.0f) * scm::_znear / d,
                  (view_pos.x + scm::_screen._screen_world_dim.x / 2.0f) * scm::_znear / d,
                  (view_pos.z - scm::_screen._screen_world_dim.y / 2.0f) * scm::_znear / d,
                  (view_pos.z + scm::_screen._screen_world_dim.y / 2.0f) * scm::_znear / d,
                  scm::_znear,
                  scm::_zfar);

        //gluPerspective(60.f, float(w)/float(h), 0.1f, 100.f);


        // restore saved matrix mode
        glMatrixMode(current_matrix_mode);
        glLoadIdentity();
        math::mat4f_t head = math::mat4f_identity;// target_it->second.transform();

        head.m12 = target_it->second.transform().m12 / 1000.0f;
        head.m13 = target_it->second.transform().m13 / 1000.0f;
        head.m14 = target_it->second.transform().m14 / 1000.0f;

        head = math::inverse(scm::_tracking_to_world_transform) * head;
        head = math::inverse(head);

        glMultMatrixf(head.mat_array);


        std::cout << std::endl;

    }

    //Sleep(200);

    std::cout << std::endl;


    // animate
    anim_angl += 0.5f;

    // on ilde just trigger a redraw
    glutPostRedisplay();
}


int main(int argc, char **argv)
{
    // the stuff that has to be done
    glutInit(&argc, argv);
    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("OpenGL - red triangle");

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
    return (0);
}