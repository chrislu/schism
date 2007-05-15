
#define NOMINMAX

#include <iostream>
#include <cassert>

#include <string>
#include <limits>

#include <scm_core/utilities/boost_warning_disable.h>

#include <boost/scoped_ptr.hpp>

// gl
#include <ogl/gl.h>
#include <ogl/utilities/error_checker.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <ogl/manipulators/trackball_manipulator.h>

#include <scm_core/time/timer.h>

#include <volume_renderer/volume_renderer_raycast_glsl.h>

// math
#include <scm_core/math/math.h>

#include <volume.h>

boost::scoped_ptr<gl::volume_renderer_raycast_glsl> _volrend_raycast;

static const float ui_float_increment = 0.1f;
static       bool  use_stencil_test   = false;

gl::trackball_manipulator _trackball_manip;

int winx = 1024;
int winy = 640;

float initx = 0;
float inity = 0;

bool lb_down = false;
bool mb_down = false;
bool rb_down = false;

float dolly_sens = 1.0f;

unsigned fbo_id       = 0;
unsigned fbo_depth_id = 0;
unsigned fbo_color_id = 0;

static bool do_inside_pass = true;
static bool draw_geometry  = true;

void render_geometry()
{
    glPushAttrib(GL_LIGHTING_BIT);
    
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);

    //glDisable(GL_COLOR_MATERIAL);

    glColor3f(0.6f, 0.6f, 0.6f);
    glPushMatrix();
        //glTranslatef(0.5f, 0.5f, 0.5f);
        glPushMatrix();
            glScalef(0.3, 0.3, 0.3);
            glTranslatef(-0.4, -0.2, 0.4);
            glRotatef(90, 1, 0, 0);
            glutSolidSphere(1.0, 20, 20);
            glTranslatef(0.8, 0.6, -0.9);
            glutSolidCube(1.0);
        glPopMatrix();
        glPushMatrix();
            glTranslatef(0.5, 0.5, 0.5);
            glutSolidCube(1.0);
        glPopMatrix();
    glPopMatrix();

    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHTING);

    glPopAttrib();
}

void render_volume()
{
    _volrend_raycast->frame(_volrend_params);
}

bool init_geometry_textures()
{
    gl::error_checker error_check;

    // color buffer texture
    glGenTextures(1, &fbo_color_id);
    if (fbo_color_id == 0) {
        std::cout << "unable to generate geometry fbo color renderbuffer texture" << std::endl;
        return (false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fbo_color_id);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, winx, winy, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    if (!error_check.ok()) {
        std::cout << "error creating geometry fbo color renderbuffer texture: ";
        std::cout << error_check.get_error_string() << std::endl;
        return (false);
    }
    else {
        std::cout << "successfully created geometry fbo color renderbuffer texture" << std::endl;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    // depth buffer texture
    glGenTextures(1, &fbo_depth_id);
    if (fbo_depth_id == 0) {
        std::cout << "unable to generate geometry fbo depth renderbuffer texture" << std::endl;
        return (false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fbo_depth_id);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH_COMPONENT24, winx, winy, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    if (!error_check.ok()) {
        std::cout << "error creating geometry fbo depth renderbuffer texture: ";
        std::cout << error_check.get_error_string() << std::endl;
        return (false);
    }
    else {
        std::cout << "successfully created geometry fbo depth renderbuffer texture" << std::endl;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    return (true);
}

bool init_geometry_fbo()
{
    // create framebuffer object
    glGenFramebuffersEXT(1, &fbo_id);
    if (fbo_id == 0) {
        std::cout << "error generating fbo id" << std::endl;
        return (false);
    }
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id);

    // attach depth texture
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_RECTANGLE_ARB, fbo_depth_id, 0);

    // attach color texture
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, fbo_color_id, 0);

    unsigned fbo_status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if (fbo_status != GL_FRAMEBUFFER_COMPLETE_EXT) {
        std::cout << "error creating fbo, framebufferstatus is not complete:" << std::endl;
        std::string error;

        switch (fbo_status) {
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:          error.assign("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:  error.assign("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:          error.assign("GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:             error.assign("GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:         error.assign("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:         error.assign("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT");break;
            case GL_FRAMEBUFFER_UNSUPPORTED_EXT:                    error.assign("GL_FRAMEBUFFER_UNSUPPORTED_EXT");break;
        }
        std::cout << "error: " << error << std::endl;
        return (false);
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    return (true);
}

void draw_geometry_color_buffer()
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);


    // push current projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    // set ortho projection
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    // switch back to modelview matrix\
    // and push current matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

        // reset current matrix to identity
        glLoadIdentity();

        // save polygon and depth buffer bit
        // to restore culling and depth mask settings later
        glPushAttrib(GL_POLYGON_BIT | GL_DEPTH_BUFFER_BIT);
            // setup and enable backface culling
            glFrontFace(GL_CCW);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);

            // disable depth writes
            glDepthMask(false);

            // enable and bind the color buffer texture rectangle
            glEnable(GL_TEXTURE_RECTANGLE_ARB);
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fbo_color_id);
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

            // draw a screen filling quad
            // (0, 0) to (1, 1) because of the set ortho projection
            //glBegin(GL_QUADS);
            //    glVertex2f(  0.0f, 0.0f);
            //    glVertex2f(  1.0f, 0.0f);
            //    glVertex2f(  1.0f, 1.0f);
            //    glVertex2f(  0.0f, 1.0f);
            //glEnd();
            glBegin(GL_TRIANGLES);
                glTexCoord2f(0.0f, 0.0f);
                glVertex2f(  0.0f, 0.0f);
                glTexCoord2f(2*winx, 0.0f);
                glVertex2f(  2.0f, 0.0f);
                glTexCoord2f(0.0f, 2*winy);
                glVertex2f(  0.0f, 2.0f);
                //glVertex2f(  0.0f, 1.0f);
            glEnd();

            // unbind and disable the color buffer texture rectangle
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
            glDisable(GL_TEXTURE_RECTANGLE_ARB);

        // restore the saved polygon and depth buffer bits
        // to reset the culling and depth mask settings
        glPopAttrib();

    // restore the saved modelview matrix
    glPopMatrix();
    // restore the saved projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    
    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

bool init_gl()
{
    // check for opengl verison 2.0 with
    // opengl shading language support
    if (!gl::is_supported("GL_VERSION_2_0")) {
        std::cout << "OpenGL 2.0 not available" << std::endl;
        std::cout << "GL_VERSION_2_0 not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "OpenGL 2.0 available" << std::endl;
        std::cout << "OpenGL Version: ";
        std::cout << (char*)glGetString(GL_VERSION) << std::endl;
    }

    if (!gl::is_supported("GL_EXT_framebuffer_object")) {
        std::cout << "GL_EXT_framebuffer_object not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "GL_EXT_framebuffer_object supported" << std::endl;
    }

    if (!gl::is_supported("GL_ARB_texture_rectangle")) {
        std::cout << "GL_ARB_texture_rectangle not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "GL_ARB_texture_rectangle supported" << std::endl;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    //glClearColor(0.3f,0.3f,0.3f,1);
    glClearColor(0.0f,0.0f,0.0f,1);

    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    glClearStencil(0);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(2);

    if (!init_geometry_textures()) {
        std::cout << "unable to create geometry fbo due to renderbuffer texture creation errors" << std::endl;
        return (false);
    }
    if (!init_geometry_fbo()) {
        return (false);
    }
    else {
        std::cout << "successfully created geometry fbo" << std::endl;
    }

    _volrend_params._geom_depth_texture_id  = fbo_depth_id;//fbo_color_id;//
    _volrend_params._screen_dimensions      = math::vec2f_t(winx, winy);


    _volrend_raycast.reset(new gl::volume_renderer_raycast_glsl());

    if (!_volrend_raycast->initialize()) {
        std::cout << "unable to initialize raycasting volume renderer" << std::endl;
        return (false);
    }

    float dif[4]    = {0.9, 0.9, 0.9, 1};
    float spc[4]    = {0.2, 0.7, 0.9, 1};
    float amb[4]    = {0.4, 0.4, 0.4, 1};
    float pos[4]    = {1,1,1,0};

    // setup light 0
    glLightfv(GL_LIGHT0, GL_SPECULAR, spc);
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
    glLightfv(GL_LIGHT0, GL_POSITION, pos);

    // define material parameters
    glMaterialfv(GL_FRONT, GL_SPECULAR, spc);
    glMaterialf(GL_FRONT, GL_SHININESS, 128.0f);
    glMaterialfv(GL_FRONT, GL_AMBIENT, amb);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, dif);

    return (true);
}

void shutdown_gl()
{
    _volrend_raycast.reset();
    glDeleteFramebuffersEXT(1, &fbo_id);
    glDeleteTextures(1, &fbo_color_id);
    glDeleteTextures(1, &fbo_depth_id);
}

void display()
{
    static scm::core::timer _timer;
    static double           _accum_time     = 0.0;
    static unsigned         _accum_count    = 0;

    _timer.start();

    // clear the color and depth buffer
    if (draw_geometry) {
        glClear(GL_DEPTH_BUFFER_BIT | /*GL_COLOR_BUFFER_BIT |*/ GL_STENCIL_BUFFER_BIT);
    }
    else {
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }

    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        _trackball_manip.apply_transform();
        
        // geometry pass
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        render_geometry();
        //_volrend_raycast->draw_outlines(_volrend_params);

        glPushAttrib(GL_POLYGON_BIT | GL_COLOR_BUFFER_BIT);
            glFrontFace(GL_CCW);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);
            glColorMask(false, false, false, false);
            _volrend_raycast->draw_bounding_volume(_volrend_params);
        glPopAttrib();

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        if (draw_geometry) {
            draw_geometry_color_buffer();
        }
    
        if (use_stencil_test) {
            glPushAttrib(GL_STENCIL_BUFFER_BIT);

            int color_mask[4];
            glGetIntegerv(GL_COLOR_WRITEMASK, color_mask);

            // write geometry stencil 
            glStencilFunc(GL_LESS, 1, 1);
            glStencilOp(GL_REPLACE, GL_KEEP, GL_KEEP);

            glColorMask(false, false, false, false);
            glEnable(GL_STENCIL_TEST);
            render_geometry();
            glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);

            glStencilFunc(GL_LESS, 0, 1);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

            // volume pass
            render_volume();

            glPopAttrib();
        }
        else {
            // volume pass
            render_volume();
        }

    // restore previously saved modelview matrix
    glPopMatrix();
    //phong_shader->unbind();

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();
    _timer.stop();

    _accum_time += _timer.get_time();
    ++_accum_count;

    if (_accum_time > 1000.0) {
        std::cout << "frame_time: " << _accum_time / static_cast<double>(_accum_count) << "msec \tfps: " << static_cast<double>(_accum_count) / (_accum_time / 1000.0) << std::endl;

        _accum_time     = 0.0;
        _accum_count    = 0;
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
    gluPerspective(60.f, float(w)/float(h), 0.1f, 100.f);


    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void keyboard(unsigned char key, int x, int y)
{
    int modifier = glutGetModifiers();

    int alt_pressed = modifier & GLUT_ACTIVE_ALT;

    switch (key) {
        // ESC key
        case 'o':
        case 'O': open_volume(); break;
        case 'X': alt_pressed != 0 ? _volrend_params._point_of_interest.x += ui_float_increment
                                   : _volrend_params._extend.x += ui_float_increment;break;
        case 'x': alt_pressed != 0 ? _volrend_params._point_of_interest.x -= ui_float_increment
                                   : _volrend_params._extend.x -= ui_float_increment;break;
        case 'Y': alt_pressed != 0 ? _volrend_params._point_of_interest.y += ui_float_increment
                                   : _volrend_params._extend.y += ui_float_increment;break;
        case 'y': alt_pressed != 0 ? _volrend_params._point_of_interest.y -= ui_float_increment
                                   : _volrend_params._extend.y -= ui_float_increment;break;
        case 'Z': alt_pressed != 0 ? _volrend_params._point_of_interest.z += ui_float_increment
                                   : _volrend_params._extend.z += ui_float_increment;break;
        case 'z': alt_pressed != 0 ? _volrend_params._point_of_interest.z -= ui_float_increment
                                   : _volrend_params._extend.z -= ui_float_increment;break;
        case 's':
        case 'S': use_stencil_test = !use_stencil_test;break;
        case 'i':
        case 'I': do_inside_pass = !do_inside_pass;
                  _volrend_raycast->do_inside_pass(do_inside_pass);
                  break;
        case 'g':
        case 'G': draw_geometry = !draw_geometry;break;
        case 27:  shutdown_gl();
                  exit (0);
                  break;
        default:;
    }
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


int main(int argc, char **argv)
{
    // the stuff that has to be done
    glutInit(&argc, argv);
    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_STENCIL);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("OpenGL - basic volume renderer");

    //glutFullScreen();

    // init the GL context
    if (!gl::initialize()) {
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


#include <scm_core/utilities/boost_warning_enable.h>
