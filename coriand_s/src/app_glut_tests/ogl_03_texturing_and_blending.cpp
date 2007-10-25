
#include <iostream>

#include <string>

#include <scm/ogl/gl.h>
#include <GL/glut.h>

#include <image_handling/image_loader.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>

scm::gl::trackball_manipulator _trackball_manip;

int winx = 1024;
int winy = 640;

float initx = 0;
float inity = 0;

bool lb_down = false;
bool mb_down = false;
bool rb_down = false;

float dolly_sens = 10.0f;

// texture objects ids
unsigned tex0_id = 0;
unsigned tex1_id = 0;

bool init_gl()
{
    // check for opengl verison 1.3 with
    // multi texture support
    if (!scm::gl::is_supported("GL_VERSION_1_3")) {
        std::cout << "GL_VERSION_1_3 not supported" << std::endl;
        return (false);
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    glClearColor(0.2f,0.2f,0.2f,1);

    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(1);

    image ps_data;

    if (!open_image("./../../../res/textures/rock_decal_gloss_512.tga", ps_data)) {
        return (false);
    }
    glGenTextures(1, &tex0_id);
    if (tex0_id == 0) {
        std::cout << "error creating texture object" << std::endl;
        return (false);
    }
    if (!load_2d_texture(tex0_id, ps_data, false)) {
        std::cout << "error uploading texture" << std::endl;
        return (false);
    }
    close_image(ps_data);

    if (!open_image("./../../../res/textures/light_map_three_color_512.jpg", ps_data)) {
        std::cout << "error open_image " << std::endl;
        return (false);
    }
    glGenTextures(1, &tex1_id);
    if (tex1_id == 0) {
        std::cout << "error creating texture object" << std::endl;
        return (false);
    }
    if (!load_2d_texture(tex1_id, ps_data, false)) {
        std::cout << "error uploading texture" << std::endl;
        return (false);
    }
    close_image(ps_data);

    return (true);
}

void draw_textured_quad()
{
    // enable texturing
    glEnable(GL_TEXTURE_2D);
    // bind texture to current opengl state
    glBindTexture(GL_TEXTURE_2D, tex0_id);

    glTranslatef(-0.5, -0.5, 0);
    // draw quad in immediate mode
    // with texture coordinates associated with every vertex
    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(0.0f, 0.0f);

      glTexCoord2f(1.0f, 0.0f);
      glVertex2f(1.0f, 0.0f);

      glTexCoord2f(1.0f, 1.0f);
      glVertex2f(1.0f, 1.0f);

      glTexCoord2f(0.0f, 1.0f);
      glVertex2f(0.0f, 1.0f);
    glEnd();

    // unbind all texture objects
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void draw_multi_textured_quad()
{
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
    // define multi texture combination mode on unit 0
    // setup to multiply color result of unit 0 with color result of unit 1
    glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_COMBINE);
    glTexEnvi(GL_TEXTURE_2D, GL_COMBINE_RGB, GL_MODULATE);


    glTranslatef(-0.5, -0.5, 0);
    // draw quad in immediate mode
    // with texture coordinates associated with every vertex
    glBegin(GL_QUADS);
      glMultiTexCoord2f(GL_TEXTURE0, 0.0f, 0.0f);
      glMultiTexCoord2f(GL_TEXTURE1, 0.0f, 0.0f);
      glVertex2f(0.0f, 0.0f);

      glMultiTexCoord2f(GL_TEXTURE0, 1.0f, 0.0f);
      glMultiTexCoord2f(GL_TEXTURE1, 1.0f, 0.0f);
      glVertex2f(1.0f, 0.0f);

      glMultiTexCoord2f(GL_TEXTURE0, 1.0f, 1.0f);
      glMultiTexCoord2f(GL_TEXTURE1, 1.0f, 1.0f);
      glVertex2f(1.0f, 1.0f);

      glMultiTexCoord2f(GL_TEXTURE0, 0.0f, 1.0f);
      glMultiTexCoord2f(GL_TEXTURE1, 0.0f, 1.0f);
      glVertex2f(0.0f, 1.0f);
    glEnd();

    // unbind all texture objects and disable texturing on texture units
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void draw_multi_pass_textured_quad()
{
    // enable texturing
    glEnable(GL_TEXTURE_2D);
    // bind texture to current opengl state
    glBindTexture(GL_TEXTURE_2D, tex0_id);

    glTranslatef(-0.5, -0.5, 0);
    // draw quad in immediate mode
    // with texture coordinates associated with every vertex
    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(0.0f, 0.0f);

      glTexCoord2f(1.0f, 0.0f);
      glVertex2f(1.0f, 0.0f);

      glTexCoord2f(1.0f, 1.0f);
      glVertex2f(1.0f, 1.0f);

      glTexCoord2f(0.0f, 1.0f);
      glVertex2f(0.0f, 1.0f);
    glEnd();

    // bind next texture to current opengl state
    glBindTexture(GL_TEXTURE_2D, tex1_id);

    // save blending states
    glPushAttrib(GL_COLOR_BUFFER_BIT);
    // enable blending
    glEnable(GL_BLEND);
    // set blending function to multiply color result with
    // framebuffer color
    glBlendFunc(GL_DST_COLOR, GL_ZERO);

    // save depth testing state
    glPushAttrib(GL_DEPTH_BUFFER_BIT);
    // set depth function to only pass if depth value
    // is equal to framebuffer content
    glDepthFunc(GL_EQUAL);

    // draw quad in immediate mode
    // with texture coordinates associated with every vertex
    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(0.0f, 0.0f);

      glTexCoord2f(1.0f, 0.0f);
      glVertex2f(1.0f, 0.0f);

      glTexCoord2f(1.0f, 1.0f);
      glVertex2f(1.0f, 1.0f);

      glTexCoord2f(0.0f, 1.0f);
      glVertex2f(0.0f, 1.0f);
    glEnd();

    // restore saved states
    glPopAttrib();
    glPopAttrib();
    
    // unbind all texture objects
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

}

void display()
{
    // clear the color and depth buffer
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        _trackball_manip.apply_transform();

        glPushMatrix();
            draw_textured_quad();
            //draw_multi_pass_textured_quad();
            //draw_multi_textured_quad();
        glPopMatrix();

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

void idle()
{
    // on ilde just trigger a redraw
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        // ESC key
        case 27: exit (0); break;
        default:;
    }
}


int main(int argc, char **argv)
{
    // the stuff that has to be done
    glutInit(&argc, argv);
    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("simple_glut");

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

    return (0);
}