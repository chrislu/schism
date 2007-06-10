
#include <iostream>

#include <set>
#include <string>

#include <scm/core/math/math.h>
#include <boost/date_time.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <scm/ogl/gl.h>
#include <GL/glut.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>
#include <scm/ogl/shader_objects/program_object.h>
#include <scm/ogl/shader_objects/shader_object.h>
#include <scm/ogl/textures/texture_2d_rect.h>
#include <scm/ogl/time/time_query.h>
#include <scm/ogl/utilities/error_checker.h>

#include <scm/core/time/high_res_timer.h>

#include <scm/core.h>
#include <scm/core/core.h>
#include <scm/core/font/face.h>
#include <scm/core/font/glyph.h>
#include <scm/core/font/face_loader.h>

#include <image_handling/image_loader.h>

boost::scoped_ptr<scm::gl::program_object>   _shader_program;
boost::scoped_ptr<scm::gl::shader_object>    _vertex_shader;
boost::scoped_ptr<scm::gl::shader_object>    _fragment_shader;

boost::scoped_ptr<scm::gl::texture_2d_rect>  _font_texture_regular;
boost::scoped_ptr<scm::gl::texture_2d_rect>  _font_texture_bold;
boost::scoped_ptr<scm::gl::texture_2d_rect>  _font_texture_italic;
boost::scoped_ptr<scm::gl::texture_2d_rect>  _font_texture_bold_italic;

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

scm::font::face     _font_face;

bool init_font_rendering()
{
    // soon to be parameters
    std::string     _font_file_name         = std::string("consola.ttf");//segoeui.ttf");//vgafix.fon");//cour.ttf");//calibri.ttf");//
    unsigned        _font_size              = 12;
    unsigned        _display_dpi_resolution = 96;

    scm::gl::error_checker _error_check;

    scm::time::high_res_timer _timer;

    glFlush();
    glFinish();
    _timer.start();

    scm::font::face_loader    _face_loader;
    _face_loader.set_font_resource_path("./../../../res/fonts/");

    if (!_face_loader.load(_font_face, _font_file_name, _font_size, _display_dpi_resolution)) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "<unnamed>::init_font_rendering(): "
                           << "error opening font file"
                           << std::endl;

        return (false);
    }

    if (_font_face.has_style(scm::font::face::regular)) {
        _font_texture_regular.reset(new scm::gl::texture_2d_rect);

        _font_texture_regular->bind();
        _font_texture_regular->tex_image(0,
                                         GL_ALPHA,
                                         _face_loader.get_current_face_texture(scm::font::face::regular)._size.x,
                                         _face_loader.get_current_face_texture(scm::font::face::regular)._size.y,
                                         GL_ALPHA,
                                         GL_UNSIGNED_BYTE,
                                         (void*)_face_loader.get_current_face_texture(scm::font::face::regular)._data.get());

        if (!_error_check.ok()) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "<unnamed>::init_font_rendering(): "
                               << "error uploading face texture: "
                               << _error_check.get_error_string()
                               << std::endl;

            return (false);
        }

        _font_texture_regular->tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        _font_texture_regular->tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        _font_texture_regular->tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        _font_texture_regular->tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    }
    else {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "<unnamed>::init_font_rendering(): "
                           << "regular style missing"
                           << std::endl;

        return (false);
    }

    if (_font_face.has_style(scm::font::face::italic)) {
        _font_texture_italic.reset(new scm::gl::texture_2d_rect);

        _font_texture_italic->bind();
        _font_texture_italic->tex_image(0,
                                         GL_ALPHA,
                                         _face_loader.get_current_face_texture(scm::font::face::italic)._size.x,
                                         _face_loader.get_current_face_texture(scm::font::face::italic)._size.y,
                                         GL_ALPHA,
                                         GL_UNSIGNED_BYTE,
                                         (void*)_face_loader.get_current_face_texture(scm::font::face::italic)._data.get());

        if (!_error_check.ok()) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "<unnamed>::init_font_rendering(): "
                               << "error uploading face texture: "
                               << _error_check.get_error_string()
                               << std::endl;

            return (false);
        }

        _font_texture_italic->tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        _font_texture_italic->tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        _font_texture_italic->tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        _font_texture_italic->tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    }
    else {
        scm::console.get() << scm::con::log_level(scm::con::info)
                           << "<unnamed>::init_font_rendering(): "
                           << "italic style missing"
                           << std::endl;
    }

    if (_font_face.has_style(scm::font::face::bold)) {
        _font_texture_bold.reset(new scm::gl::texture_2d_rect);

        _font_texture_bold->bind();
        _font_texture_bold->tex_image(0,
                                         GL_ALPHA,
                                         _face_loader.get_current_face_texture(scm::font::face::bold)._size.x,
                                         _face_loader.get_current_face_texture(scm::font::face::bold)._size.y,
                                         GL_ALPHA,
                                         GL_UNSIGNED_BYTE,
                                         (void*)_face_loader.get_current_face_texture(scm::font::face::bold)._data.get());

        if (!_error_check.ok()) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "<unnamed>::init_font_rendering(): "
                               << "error uploading face texture: "
                               << _error_check.get_error_string()
                               << std::endl;

            return (false);
        }

        _font_texture_bold->tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        _font_texture_bold->tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        _font_texture_bold->tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        _font_texture_bold->tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    }
    else {
        scm::console.get() << scm::con::log_level(scm::con::info)
                           << "<unnamed>::init_font_rendering(): "
                           << "bold style missing"
                           << std::endl;
    }

    if (_font_face.has_style(scm::font::face::bold_italic)) {
        _font_texture_bold_italic.reset(new scm::gl::texture_2d_rect);

        _font_texture_bold_italic->bind();
        _font_texture_bold_italic->tex_image(0,
                                         GL_ALPHA,
                                         _face_loader.get_current_face_texture(scm::font::face::bold_italic)._size.x,
                                         _face_loader.get_current_face_texture(scm::font::face::bold_italic)._size.y,
                                         GL_ALPHA,
                                         GL_UNSIGNED_BYTE,
                                         (void*)_face_loader.get_current_face_texture(scm::font::face::bold_italic)._data.get());

        if (!_error_check.ok()) {
            scm::console.get() << scm::con::log_level(scm::con::error)
                               << "<unnamed>::init_font_rendering(): "
                               << "error uploading face texture: "
                               << _error_check.get_error_string()
                               << std::endl;

            return (false);
        }

        _font_texture_bold_italic->tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        _font_texture_bold_italic->tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        _font_texture_bold_italic->tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        _font_texture_bold_italic->tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    }
    else {
        scm::console.get() << scm::con::log_level(scm::con::info)
                           << "<unnamed>::init_font_rendering(): "
                           << "bold_italic style missing"
                           << std::endl;
    }

    glFlush();
    glFinish();

    _timer.stop();

    std::stringstream output;

    output.precision(2);
    output << std::fixed
           << "font setup time: " << scm::time::to_milliseconds(_timer.get_time()) << "msec" << std::endl;

    scm::console.get() << output.str();

    return (true);
}

bool init_gl()
{
    // check for opengl verison 2.0 with
    // opengl shading language support
    if (!scm::ogl.get().is_supported("GL_VERSION_2_0")) {
        std::cout << "GL_VERSION_2_0 not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "OpenGL 2.0 supported, OpenGL Version: " << (char*)glGetString(GL_VERSION) << std::endl;
    }
    if (!scm::gl::time_query::is_supported()) {
        std::cout << "scm::gl::time_query not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "scm::gl::time_query available" << std::endl;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    glClearColor(0, 0, 0, 1);
    //glClearColor(1, 1, 1, 1);
    //glClearColor(0.2f,0.2f,0.2f,1);
    //glClearColor(0.8f,0.8f,0.8f,1);

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
    if (!_vertex_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/texture_vertex_program.glsl")) {
        std::cout << "Error loadong vertex shader:" << std::endl;
        return (false);
    }
    if (!_fragment_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/texture_fragment_program.glsl")) {
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

    image ps_data;

    // load texture image
    if (!open_image("./../../../res/textures/rock_decal_gloss_512.tga", ps_data)) {
        return (false);
    }
    // generate texture object id
    glGenTextures(1, &tex0_id);
    if (tex0_id == 0) {
        std::cout << "error creating texture object" << std::endl;
        return (false);
    }
    // load image as texture to texture object
    if (!load_2d_texture(tex0_id, ps_data, false)) {
        std::cout << "error uploading texture" << std::endl;
        return (false);
    }
    // delete image
    close_image(ps_data);

    // load texture image
    if (!open_image("./../../../res/textures/rock_normal_512.tga", ps_data)) {
        std::cout << "error open_image " << std::endl;
        return (false);
    }
    // generate texture object id
    glGenTextures(1, &tex1_id);
    if (tex1_id == 0) {
        std::cout << "error creating texture object" << std::endl;
        return (false);
    }
    // load image as texture to texture object
    if (!load_2d_texture(tex1_id, ps_data, false)) {
        std::cout << "error uploading texture" << std::endl;
        return (false);
    }
    // delete image
    close_image(ps_data);

    if (!init_font_rendering()) {
        std::cout << "error initializing font rendering engine" << std::endl;
        return (false);
    }


    float dif[4]    = {0.7, 0.7, 0.7, 1};
    float spc[4]    = {0.2, 0.7, 0.9, 1};
    float amb[4]    = {0.1, 0.1, 0.1, 1};
    float pos[4]    = {1,1,1,0};

    // setup light 0
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);

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

void draw_font_image()
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    // setup orthogonal projection
    // setup as 1:1 pixel mapping
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winx, 0, winy, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1, 1, 1);

    // save states which we change in here
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glPushAttrib(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    // setup blending
    glPushAttrib(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    _font_texture_regular->bind();

    // draw a quad in the size of the font texture
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(  0.0f, 0.0f);
        glTexCoord2f(_font_texture_regular->get_width(), 0.0f);
        glVertex2f(  _font_texture_regular->get_width(), 0.0f);
        glTexCoord2f(_font_texture_regular->get_width(), _font_texture_regular->get_height());
        glVertex2f(  _font_texture_regular->get_width(), _font_texture_regular->get_height());
        glTexCoord2f(0.0f, _font_texture_regular->get_height());
        glVertex2f(  0.0f, _font_texture_regular->get_height());
    glEnd();

    _font_texture_regular->unbind();

    // restore saved states
    glPopAttrib();
    glPopAttrib();
    glPopAttrib();

    // restore previous projection
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void draw_quad(const math::vec2i_t& lower_left,
               const math::vec2i_t& upper_right,
               const math::vec2i_t& tex_lower_left,
               const math::vec2i_t& tex_upper_right)
{
    glBegin(GL_QUADS);
        glTexCoord2f(tex_lower_left.x,  tex_lower_left.y);
        glVertex2f(  lower_left.x,      lower_left.y);

        glTexCoord2f(tex_upper_right.x, tex_lower_left.y);
        glVertex2f(  upper_right.x,     lower_left.y);

        glTexCoord2f(tex_upper_right.x, tex_upper_right.y);
        glVertex2f(  upper_right.x,     upper_right.y);

        glTexCoord2f(tex_lower_left.x,  tex_upper_right.y);
        glVertex2f(  lower_left.x,      upper_right.y);
    glEnd();
}

void draw_string(scm::font::face::style_type style,
                 const math::vec2i_t& position,
                 const std::string& stuff_that_matters)
{
    using namespace scm::font;

    scm::gl::texture_2d_rect* cur_style_tex;

    switch (style) {
        case face::regular:
            cur_style_tex = _font_texture_regular.get();
            break;
        case face::italic:
            if (_font_face.has_style(style)) {
                cur_style_tex = _font_texture_italic.get();
            }
            else {
                cur_style_tex = _font_texture_regular.get();
            }
            break;
        case face::bold:
            if (_font_face.has_style(style)) {
                cur_style_tex = _font_texture_bold.get();
            }
            else {
                cur_style_tex = _font_texture_regular.get();
            }
            break;
        case face::bold_italic:
            if (_font_face.has_style(style)) {
                cur_style_tex = _font_texture_bold_italic.get();
            }
            else {
                cur_style_tex = _font_texture_regular.get();
            }
            break;
    };

    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    // setup orthogonal projection
    // setup as 1:1 pixel mapping
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winx, 0, winy, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // save states which we change in here
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glPushAttrib(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    // setup blending
    glPushAttrib(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    cur_style_tex->bind();

    math::vec2i_t current_pos = position;

    unsigned prev = 0;

    for (std::string::const_iterator c = stuff_that_matters.begin();
         c != stuff_that_matters.end();
         ++c) {
        const scm::font::glyph& cur_glyph = _font_face.get_glyph(*c, style);

        if (prev) {
            current_pos.x += _font_face.get_kerning(prev, *c, style);
        }

        // draw shadow first
        //glColor4f(0, 0, 0, 1);
        //draw_quad(current_pos + math::vec2i_t(1,-1) + cur_glyph._bearing,
        //          current_pos + math::vec2i_t(1,-1) + cur_glyph._bearing + cur_glyph._tex_upper_right - cur_glyph._tex_lower_left,
        //          cur_glyph._tex_lower_left,
        //          cur_glyph._tex_upper_right);

        // draw glyph
        glColor4f(1, 1, 1, 1);
        draw_quad(current_pos + cur_glyph._bearing,
                  current_pos + cur_glyph._bearing + cur_glyph._tex_upper_right - cur_glyph._tex_lower_left,
                  cur_glyph._tex_lower_left,
                  cur_glyph._tex_upper_right);

        // advance the position
        current_pos.x += cur_glyph._advance;

        // remember just drawn glyph for kerning
        prev = *c;
    }
    
    cur_style_tex->unbind();

    // restore saved states
    glPopAttrib();
    glPopAttrib();
    glPopAttrib();

    // restore previous projection
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void display()
{
    static double           _accum_time     = 0.0;
    static double           _gl_accum_time  = 0.0;
    static unsigned         _accum_count    = 0;

    static scm::time::high_res_timer _timer;
    static scm::gl::time_query            _gl_timer;

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();

    //glFlush();
    //glFinish();
    _timer.stop();

    _timer.start();
    _gl_timer.start();

    // clear the color and depth buffer
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        _trackball_manip.apply_transform();

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

        // bind shader program to current opengl state
        _shader_program->bind();
        // set program parameters
        // set the sampler parameters to the particular texture unit number
        _shader_program->set_uniform_1i("_diff_gloss", 0);
        _shader_program->set_uniform_1i("_normal", 1);
        // set shininess material parameter directly to unfiform parameter
        _shader_program->set_uniform_1f("_shininess", 128.0f);

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

        // unbind shader program
        _shader_program->unbind();

        // unbind all texture objects and disable texturing on texture units
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

    // restore previously saved modelview matrix
    glPopMatrix();

    //draw_font_image();

    // mjM hallo Welt! To ro  :\  .\ []{}|||~!@#$%^&*()_+;:<>/|'"~` 0123456789
    draw_string(scm::font::face::regular,
                math::vec2i_t(20, winy - 40),
                "mjM hallo Welt! To ro  :\\  .\\ []{}|||~!@#$%^&*()_+;:<>/|'\"~` 0123456789");
    draw_string(scm::font::face::italic,
                math::vec2i_t(20, winy - 40 - _font_face.get_line_spacing()),
                "mjM hallo Welt! To ro  :\\  .\\ []{}|||~!@#$%^&*()_+;:<>/|'\"~` 0123456789");
    draw_string(scm::font::face::bold,
                math::vec2i_t(20, winy - 40 - 2 * _font_face.get_line_spacing()),
                "mjM hallo Welt! To ro  :\\  .\\ []{}|||~!@#$%^&*()_+;:<>/|'\"~` 0123456789");
    draw_string(scm::font::face::bold_italic,
                math::vec2i_t(20, winy - 40 - 3 * _font_face.get_line_spacing()),
                "mjM hallo Welt! To ro  :\\  .\\ []{}|||~!@#$%^&*()_+;:<>/|'\"~` 0123456789");

    _gl_timer.stop();

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();

    //glFlush();
    //glFinish();
    _timer.stop();

    _gl_timer.collect_result();

    _accum_time += scm::time::to_milliseconds(_timer.get_time());
    _gl_accum_time += scm::time::to_milliseconds(_gl_timer.get_time());
    ++_accum_count;

    if (_accum_time > 1000.0) {
        std::stringstream   output;

        output.precision(2);
        output << std::fixed << "frame_time: " << _accum_time / static_cast<double>(_accum_count) << "msec \t"
                             << "gl time: " << _gl_accum_time / static_cast<double>(_accum_count) << "msec \t"
                             << "fps: " << static_cast<double>(_accum_count) / (_accum_time / 1000.0)
                             << std::endl;

        scm::console.get() << output.str();

        _accum_time     = 0.0;
        _gl_accum_time  = 0.0;
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
    int width;
    int height;
    bool fullscreen;

    try {
        boost::program_options::options_description  cmd_options("program options");

        cmd_options.add_options()
            ("help", "show this help message")
            ("width", boost::program_options::value<int>(&width)->default_value(1024), "output width")
            ("height", boost::program_options::value<int>(&height)->default_value(640), "output height")
            ("fullscreen", boost::program_options::value<bool>(&fullscreen)->default_value(false), "run in fullscreen mode");

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

    return (0);
}