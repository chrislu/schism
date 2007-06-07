
#include <iostream>

#include <string>

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include <scm/ogl/gl.h>
#include <GL/glut.h>

#include <IL/il.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>
#include <scm/ogl/shader_objects/program_object.h>
#include <scm/ogl/shader_objects/shader_object.h>
#include <scm/ogl/utilities/error_checker.h>

#include <obj_handling/obj_file.h>
#include <obj_handling/obj_loader.h>

#include <image_handling/image_loader.h>

boost::scoped_ptr<scm::gl::program_object>  _fbo_shader_program;
boost::scoped_ptr<scm::gl::shader_object>   _fbo_vertex_shader;
boost::scoped_ptr<scm::gl::shader_object>   _fbo_fragment_shader;

scm::gl::trackball_manipulator _trackball_manip;

// from commandline set!
static int winx;
static int winy;

float initx = 0;
float inity = 0;

bool lb_down = false;
bool mb_down = false;
bool rb_down = false;

float dolly_sens = 1.0f;

// some constants
static const std::string geometry_obj_file      = std::string("./../../../res/objects/some_objects.obj");
static const std::string correction_image_name  = std::string("./../../../res/textures/geometry_correction_image.png");

// texture objects ids
unsigned tex_correction_img     = 0;

unsigned fbo_id                 = 0;
unsigned fbo_depth_id           = 0;
unsigned fbo_color_id           = 0;

// flag for wireframe rendering of the geometry
bool     draw_wireframe         = false;

// the obj file object
scm::wavefront_model obj_f;

void render_wavefront_obj_immediate_mode()
{
    // retrieve current polygon mode
    GLint current_poly_mode[2];
    glGetIntegerv(GL_POLYGON_MODE, current_poly_mode);
    // set polygon mode to wireframe rendering
    glPolygonMode(GL_FRONT_AND_BACK, draw_wireframe ? GL_LINE : GL_FILL);
    glShadeModel(GL_SMOOTH);

    glColor3f(0.2f, 0.5f, 0.2f);

    scm::wavefront_model::object_container::iterator     cur_obj_it;
    scm::wavefront_object::group_container::iterator     cur_grp_it;

    glPushMatrix();
        glBegin(GL_TRIANGLES);

        for (cur_obj_it = obj_f._objects.begin(); cur_obj_it != obj_f._objects.end(); ++cur_obj_it) {
            for (cur_grp_it = cur_obj_it->_groups.begin(); cur_grp_it != cur_obj_it->_groups.end(); ++cur_grp_it) {
                for (unsigned i = 0; i < cur_grp_it->_num_tri_faces; ++i) {
                    for (unsigned k = 0; k < 3; ++k) {
                        glNormal3fv(obj_f._normals[cur_grp_it->_tri_faces[i]._normals[k] - 1].vec_array);
                        glVertex3fv(obj_f._vertices[cur_grp_it->_tri_faces[i]._vertices[k] - 1].vec_array);
                    }
                }
            }
        }
        glEnd();
    glPopMatrix();

    // restore saved polygon mode
    glPolygonMode(GL_FRONT, current_poly_mode[0]);
    glPolygonMode(GL_BACK,  current_poly_mode[1]);

}

void render_geometry()
{
    glPushAttrib(GL_LIGHTING_BIT);
    
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);

    render_wavefront_obj_immediate_mode();

    glPopAttrib();
}

bool init_geometry_textures()
{
    scm::gl::error_checker error_check;

    // color buffer texture
    glGenTextures(1, &fbo_color_id);
    if (fbo_color_id == 0) {
        std::cout << "unable to generate geometry fbo color renderbuffer texture" << std::endl;
        return (false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fbo_color_id);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);//GL_NEAREST);//
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);//GL_NEAREST);//
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

bool load_correction_image()
{
    image ps_data;

    // load texture image
    if (!open_image(correction_image_name, ps_data)) {
        std::cout << "unable to open correction image file: " << correction_image_name << std::endl;
        return (false);
    }
    // generate texture object id
    glGenTextures(1, &tex_correction_img);
    if (tex_correction_img == 0) {
        std::cout << "error creating texture object for correction image" << std::endl;
        return (false);
    }

    if (ps_data._image_type != IL_UNSIGNED_BYTE) {
        std::cout << "error: unsupported image type, IL_UNSIGNED_BYTE expected" << std::endl;
        return (false);
    }

    if (ps_data._width != winx || ps_data._height != winy) {
        std::cout << "correction image dimensions differ from output resolution!" << std::endl;
        std::cout << "correction image resolution: " << ps_data._width << "x" << ps_data._height << std::endl;
        std::cout << "output resolution:           " << winx << "x" << winy << std::endl;
        return (false);
    }
    else {
        std::cout << "correction image resolution: " << ps_data._width << "x" << ps_data._height << std::endl;
    }

    GLenum source_type = GL_UNSIGNED_BYTE;
    GLenum internal_format;
    GLenum source_format;

    switch (ps_data._image_format) {
        case IL_LUMINANCE:          internal_format = GL_LUMINANCE; source_format = GL_LUMINANCE; break;
        case IL_LUMINANCE_ALPHA:    internal_format = GL_LUMINANCE_ALPHA; source_format = GL_LUMINANCE_ALPHA; break;
        case IL_BGR:                internal_format = GL_RGB; source_format = GL_BGR; break;
        case IL_BGRA:               internal_format = GL_RGBA; source_format = GL_BGRA; break;
        case IL_RGB:                internal_format = GL_RGB; source_format = GL_RGB; break;
        case IL_RGBA:               internal_format = GL_RGBA; source_format = GL_RGBA; break;
        default: return (false);
    }

    ilBindImage(ps_data._id);

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_correction_img);
    glEnable(GL_TEXTURE_RECTANGLE_ARB);

    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, internal_format, ps_data._width, ps_data._height, 0, source_format, source_type, (void*)ilGetData());
    if (glGetError() != GL_NO_ERROR) {
        std::cout << "error uploading correction image texture texture" << std::endl;
        return (false);
    }
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);//GL_LINEAR);//
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);//GL_LINEAR);//

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    ilBindImage(0);

    close_image(ps_data);

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

bool init_fbo_correction_shader_program()
{
    _fbo_shader_program.reset(new scm::gl::program_object());
    _fbo_vertex_shader.reset(new scm::gl::shader_object(GL_VERTEX_SHADER));
    _fbo_fragment_shader.reset(new scm::gl::shader_object(GL_FRAGMENT_SHADER));

    // load shader code from files
    if (!_fbo_vertex_shader->set_source_code_from_file("./../../../src/app_geometry_correction/shader/geom_correct_vert.glsl")) {
        std::cout << "Error loading vertex shader file" << std::endl;
        return (false);
    }
    if (!_fbo_fragment_shader->set_source_code_from_file("./../../../src/app_geometry_correction/shader/geom_correct_frag.glsl")) {
        std::cout << "Error loading frament shader file" << std::endl;
        return (false);
    }

    // compile shaders
    if (!_fbo_vertex_shader->compile()) {
        std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
        std::cout << _fbo_vertex_shader->get_compiler_output() << std::endl;
        return (false);
    }
    if (!_fbo_fragment_shader->compile()) {
        std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
        std::cout << _fbo_fragment_shader->get_compiler_output() << std::endl;
        return (false);
    }

    // attatch shaders to program object
    if (!_fbo_shader_program->attach_shader(*_fbo_vertex_shader)) {
        std::cout << "unable to attach vertex shader to program object:" << std::endl;
        return (false);
    }
    if (!_fbo_shader_program->attach_shader(*_fbo_fragment_shader)) {
        std::cout << "unable to attach fragment shader to program object:" << std::endl;
        return (false);
    }

    // link program object
    if (!_fbo_shader_program->link()) {
        std::cout << "Error linking program - linker output:" << std::endl;
        std::cout << _fbo_shader_program->get_linker_output() << std::endl;
       return (false);
    }

    return (true);
}

bool init_gl()
{
    // check for opengl verison 2.0 with
    // opengl shading language support
    if (!scm::gl::is_supported("GL_VERSION_2_0")) {
        std::cout << "OpenGL 2.0 not available" << std::endl;
        std::cout << "GL_VERSION_2_0 not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "OpenGL 2.0 available" << std::endl;
        std::cout << "OpenGL Version: ";
        std::cout << (char*)glGetString(GL_VERSION) << std::endl;
    }

    if (!scm::gl::is_supported("GL_EXT_framebuffer_object")) {
        std::cout << "GL_EXT_framebuffer_object not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "GL_EXT_framebuffer_object supported" << std::endl;
    }

    if (!scm::gl::is_supported("GL_ARB_texture_rectangle")) {
        std::cout << "GL_ARB_texture_rectangle not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "GL_ARB_texture_rectangle supported" << std::endl;
    }

    std::cout << (char*)glGetString(GL_VERSION) << std::endl;

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    // glClearColor(0.2f,0.2f,0.2f,1);
    glClearColor(1.0f,1.0f,1.0f,1);

    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(6);

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
    if (!load_correction_image()) {
        return (false);
    }
    else {
        std::cout << "successfully loaded correction image: " << correction_image_name << std::endl;
    }
    if (!init_fbo_correction_shader_program()) {
        return (false);
    }
    else {
        std::cout << "successfully initialized fbo correction shader program" << std::endl;
    }
    if (!scm::open_obj_file(geometry_obj_file, obj_f)) {
        std::cout << "failed loading obj file: " << geometry_obj_file << std::endl;
        return (false);
    }
    else {
        std::cout << "successfully loaded obj file: " << geometry_obj_file << std::endl;
    }


    // setup some light for the geometry
    float dif[4]    = {0.7, 0.7, 0.7, 1};
    float spc[4]    = {0.2, 0.7, 0.9, 1};
    float amb[4]    = {0.1, 0.1, 0.1, 1};
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
    glDeleteFramebuffersEXT(1, &fbo_id);
    glDeleteTextures(1, &fbo_color_id);
    glDeleteTextures(1, &fbo_depth_id);
    glDeleteTextures(1, &tex_correction_img);
}

void render_to_fbo()
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id);

    // clear the color and depth buffer
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        _trackball_manip.apply_transform();

        render_geometry();

    // restore previously saved modelview matrix
    glPopMatrix();

    // needed on pre g80, no idea why
    // without it there is flickering on the output, when reading
    // from the fbo textures in a shader
    glFinish();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void draw_fbo_image()
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    // push current projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    // set ortho projection
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    // switch back to modelview matrix
    // and push current matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

        // reset current matrix to identity
        glLoadIdentity();

        // save polygonbit to restore culling settings later
        glPushAttrib(GL_POLYGON_BIT);
            // setup and enable backface culling
            glFrontFace(GL_CCW);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);

            // enable and bind the color buffer texture rectangle
            // to texture unit 0
            glActiveTexture(GL_TEXTURE0);
            glEnable(GL_TEXTURE_RECTANGLE_ARB);
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fbo_color_id);

            // enable and bind the correction image texture rectangle
            // to texture unit 1
            glActiveTexture(GL_TEXTURE1);
            glEnable(GL_TEXTURE_RECTANGLE_ARB);
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_correction_img);

            // bind the fbo shader program and set its parameters
            _fbo_shader_program->bind();
            _fbo_shader_program->set_uniform_1i("_geometry_image", 0);
            _fbo_shader_program->set_uniform_1i("_correction_image", 1);

            // draw a screen filling quad
            // (0, 0) to (1, 1) because of the set ortho projection
#if 0
            glBegin(GL_QUADS);
                glVertex2f(  0.0f, 0.0f);
                glVertex2f(  1.0f, 0.0f);
                glVertex2f(  1.0f, 1.0f);
                glVertex2f(  0.0f, 1.0f);
            glEnd();
#else
            glBegin(GL_TRIANGLES);
                //glTexCoord2f(0.0f, 0.0f);
                glVertex2f(  0.0f, 0.0f);
                //glTexCoord2f(2*winx, 0.0f);
                glVertex2f(  2.0f, 0.0f);
                //glTexCoord2f(0.0f, 2*winy);
                glVertex2f(  0.0f, 2.0f);
                //glVertex2f(  0.0f, 1.0f);
            glEnd();
#endif
            // unbind the fbo shader program
            _fbo_shader_program->unbind();

            // unbind and disable the correction image texture rectangle
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
            glDisable(GL_TEXTURE_RECTANGLE_ARB);

            // unbind and disable the color buffer texture rectangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
            glDisable(GL_TEXTURE_RECTANGLE_ARB);

        // restore the saved polygonbits to reset the culling settings
        glPopAttrib();

    // restore the saved modelview matrix
    glPopMatrix();
    // restore the saved projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    
    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void display()
{
    // only need to clear depth component, because
    // in the fullscreen quad pass with the fbo image
    // we overdraw the whole viewport
    glClear(GL_DEPTH_BUFFER_BIT /*| GL_COLOR_BUFFER_BIT*/);

    render_to_fbo();
    draw_fbo_image();

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
        case 27:    shutdown_gl();
                    exit (0);
                    break;
        case 'w':
        case 'W':   draw_wireframe = !draw_wireframe;break;
        default:;
    }
}


int main(int argc, char **argv)
{
    int width;
    int height;


    try {
        boost::program_options::options_description  cmd_options("program options");

        cmd_options.add_options()
            ("help", "show this help message")
            ("width", boost::program_options::value<int>(&width)->default_value(1024), "output width")
            ("height", boost::program_options::value<int>(&height)->default_value(640), "output height");

        boost::program_options::variables_map       command_line;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, cmd_options), command_line);
        boost::program_options::notify(command_line);

        if (command_line.count("help")) {
            std::cout << "usage: " << std::endl;
            std::cout << cmd_options;
            return (0);
        }
        winx = width;
        winy = height;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return (-1);
    }

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



