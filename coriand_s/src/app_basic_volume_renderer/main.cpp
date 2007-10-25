
#define NOMINMAX

#include <iostream>
#include <cassert>

#include <string>
#include <limits>

#include <scm/core/utilities/boost_warning_disable.h>

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

// gl
#include <scm/ogl/gl.h>
#include <GL/glut.h>
#include <scm/ogl/utilities/error_checker.h>
#include <scm/ogl/utilities/axes_compass.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>
#include <scm/ogl/time/time_query.h>
#include <scm/ogl/gui/console_renderer.h>

#include <scm/ogl/shader_objects/program_object.h>
#include <scm/ogl/shader_objects/shader_object.h>

#include <scm/core/time/high_res_timer.h>

#include <scm/ogl/font/face.h>
#include <scm/ogl/gui/font_renderer.h>
#include <scm/ogl/font/face_loader.h>

#include <volume_renderer/volume_renderer_raycast_glsl.h>
#include <volume_renderer/volume_renderer_crossplanes_glsl.h>

#include <scm/core.h>
#include <scm/core/math/math.h>
#include <scm/core/utilities/foreach.h>

#include <volume.h>
#include <geometry.h>

boost::scoped_ptr<gl::volume_renderer_raycast_glsl> _volrend_raycast;
boost::scoped_ptr<gl::volume_renderer_crossplanes_glsl> _volrend_cross_planes;

boost::scoped_ptr<scm::gl::gui::console_renderer>   _console_rend;

boost::scoped_ptr<scm::gl::program_object>   _shader_program;
boost::scoped_ptr<scm::gl::shader_object>    _vertex_shader;
boost::scoped_ptr<scm::gl::shader_object>    _fragment_shader;

typedef enum
{
    console_full    = 1,
    console_brief   = 2,
    console_hide    = 3
} con_mode;

con_mode _con_mode = console_brief;

unsigned        _show_image     = 0;
// 0 - final rendering
// 1 - sc color image
// 2 - sc depth image
// 3 - fc stencil image
// 4 - fc depth image
// 5 - fc color image

bool            _high_quality_volume = false;
float           _near_plane          = 0.1;

static const float ui_float_increment = 0.1f;
static       bool  use_stencil_test   = false;
static       int   use_vcal           = 0;

bool _anim_enabled = false;
float _anim_step = 1.0f;
scm::gl::trackball_manipulator _trackball_manip;

int winx = 0;
int winy = 0;

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

void con_mode_changed()
{
    if (_con_mode == console_full) {
        _console_rend->position(math::vec2i_t(0, 0));
        _console_rend->size(math::vec2i_t(winx, winy));
    }
    else if (_con_mode == console_brief) {
        _console_rend->position(math::vec2i_t(0, winy - 100));
        _console_rend->size(math::vec2i_t(winx, 100));
    }
    else if (_con_mode == console_hide) {
    }

}
std::vector<math::vec3f_t> colors;
bool init_font_rendering()
{
    //std::cout << "fmid " << scm::gl::gl_font_manager_id << std::endl;
    // soon to be parameters
    std::string     _font_file_name         = std::string("consola.ttf");//segoeui.ttf");//calibri.ttf");//vgafix.fon");//cour.ttf");//
    unsigned        _font_size              = 10;
    unsigned        _display_dpi_resolution = 96;
    scm::gl::face_ptr          _font_face;

    scm::time::high_res_timer _timer;

    glFinish();
    _timer.start();

    scm::gl::face_loader _face_loader;
    _face_loader.set_font_resource_path("./../../../res/fonts/");

    _font_face = _face_loader.load(_font_file_name, _font_size, _display_dpi_resolution);
    
    if (!_font_face) {
        scm::console.get() << scm::con::log_level(scm::con::error)
                           << "<unnamed>::init_font_rendering(): "
                           << "error opening font file"
                           << std::endl;

        return (false);
    }

    glFinish();

    _timer.stop();

    _console_rend.reset(new scm::gl::gui::console_renderer());
    _console_rend->position(math::vec2i_t(0, 0));
    _console_rend->size(math::vec2i_t(winx, winy));
    _console_rend->font(_font_face);

    _console_rend->connect(scm::console.get().out_stream());

    std::stringstream output;

    output.precision(2);
    output << std::fixed
           << "font setup time: " << scm::time::to_milliseconds(_timer.get_time()) << "msec" << std::endl;

    scm::console.get() << output.str();

    return (true);
}

void render_geometry()
{
    static float anim_steps = 0.0f;

    anim_steps = anim_steps + _volrend_params._anim_speed * _volrend_params._last_frame_time / 1000.0f;

    float anim_step;
    if (_anim_enabled) {
        anim_step = math::cos(anim_steps * 2.0f * math::pi_f) * 0.5f + 0.5f;
    }
    else {
        anim_step = _anim_step;//params._anim_step;
    }

    glPushAttrib(GL_LIGHTING_BIT);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    
#if 0
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glDisable(GL_COLOR_MATERIAL);
    glColor3f(0.6f, 0.6f, 0.6f);
    glPushMatrix();
        glTranslatef(0.5f, 0.5f, 0.5f);
        glPushMatrix();
            glScalef(0.3, 0.3, 0.3);
            glTranslatef(-0.4, -0.2, 0.4);
            glRotatef(90, 1, 0, 0);
            glutSolidSphere(1.0, 20, 20);
            glTranslatef(0.8, 0.6, -0.9);
            //glutSolidCube(1.0);
        glPopMatrix();
        glPushMatrix();
            glTranslatef(0.5, 0.5, 0.5);
            glutSolidCube(1.0);
        glPopMatrix();
    glPopMatrix();
#else

    if (   _show_image == 0
        || _show_image == 1
        || _show_image == 2) {
        _volrend_cross_planes->frame(_volrend_params);
    }

#if 0
    glActiveTexture(GL_TEXTURE0);
    _volrend_params._uncertainty_volume_texture.bind();


    _shader_program->bind();

    math::mat_glf_t modelview;
    math::get_gl_matrix(GL_MODELVIEW_MATRIX, modelview);

    math::mat_glf_t vertex_vol_aspect_scale = math::mat4f_identity;

    vertex_vol_aspect_scale.scale(
        _volrend_params._aspect.x,
        _volrend_params._aspect.y,
        _volrend_params._aspect.z);

    glColor3f(0.6f, 0.6f, 0.6f);

#endif
    glPushMatrix();

    unsigned c = 0;

#if 0
    foreach(const geometry& geom, _geometries) {
        glColor3f(colors[c].x, colors[c].y, colors[c].z);
        ++c;
        math::mat_glf_t vertex_to_volume_unit_transform       = math::mat4f_identity;
        math::mat_glf_t vertex_to_volume_transform            = vertex_vol_aspect_scale;

        vertex_to_volume_unit_transform.scale(
            1.0f /(_data_properties._vol_desc._volume_aspect.x *float(_data_properties._vol_desc._data_dimensions.x)),
            1.0f /(_data_properties._vol_desc._volume_aspect.y *float(_data_properties._vol_desc._data_dimensions.y)),
            1.0f /(_data_properties._vol_desc._volume_aspect.z *float(_data_properties._vol_desc._data_dimensions.z)));
        
        vertex_to_volume_unit_transform.translate(
            - _data_properties._vol_desc._volume_origin.x,
            - _data_properties._vol_desc._volume_origin.y,
            - _data_properties._vol_desc._volume_origin.z);
        
        vertex_to_volume_unit_transform.translate(
            geom._desc._geometry_origin.x,
            geom._desc._geometry_origin.y,
            geom._desc._geometry_origin.z);
        
        vertex_to_volume_unit_transform.scale(
            geom._desc._geometry_scale.x,
            geom._desc._geometry_scale.y,
            geom._desc._geometry_scale.z);

        vertex_to_volume_transform *= vertex_to_volume_unit_transform;
        math::mat_glf_t norm_mat =  modelview * vertex_to_volume_transform;
        norm_mat =  math::transpose(math::inverse(norm_mat));
        
        _shader_program->set_uniform_1i("_unc_texture", 0);
        _shader_program->set_uniform_1f("_anim_step", anim_step);
        _shader_program->set_uniform_matrix_4fv("_vert2unit", 1, false, vertex_to_volume_unit_transform.mat_array);
        _shader_program->set_uniform_matrix_4fv("_vert2vol",  1, false, vertex_to_volume_transform.mat_array);
        _shader_program->set_uniform_matrix_4fv("_vert2vol_it",  1, false, norm_mat.mat_array);
#endif

    if (   _show_image == 0
        || _show_image == 1
        || _show_image == 2) {

        const geometry& geom = _geometries[0];{

            glEnable(GL_NORMALIZE);
            glPushMatrix();
            glTranslatef(.1,
                         0,
                         0.2);

            glTranslatef(geom._desc._geometry_origin.x,
                         geom._desc._geometry_origin.y,
                         geom._desc._geometry_origin.z);

            glScalef(0.7, 0.7, 0.7);

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
    }
    if (   _show_image == 0
        || _show_image == 3
        || _show_image == 4
        || _show_image == 5) {

        const geometry& geom = _geometries[1];{

            glEnable(GL_NORMALIZE);
            glPushMatrix();
            glTranslatef(.1,
                         0,
                         0.2);

            glTranslatef(geom._desc._geometry_origin.x,
                         geom._desc._geometry_origin.y,
                         geom._desc._geometry_origin.z);

            glScalef(0.7, 0.7, 0.7);

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
    }
            //geom._vbo->bind();
            //geom._vbo->draw_elements();
            //geom._vbo->unbind();
//    }
    glPopMatrix();
    _shader_program->unbind();
    _volrend_params._volume_texture.unbind();
#endif
    glDisable(GL_LIGHT0);
    //glDisable(GL_LIGHTING);

    glPopAttrib();
}

void render_volume()
{
    _volrend_raycast->frame(_volrend_params);
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

    // save polygon and depth buffer bit
    // to restore culling and depth mask settings later
    glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT);

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

    // restore the saved polygon and depth buffer bits
    // to reset the culling and depth mask settings
    glPopAttrib();

    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

bool init_gl()
{
    // check for opengl verison 2.0 with
    // opengl shading language support
    if (!scm::ogl.get().is_supported("GL_VERSION_2_0")) {
        std::cout << "OpenGL 2.0 not available" << std::endl;
        std::cout << "GL_VERSION_2_0 not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "OpenGL 2.0 available" << std::endl;
        std::cout << "OpenGL Version: ";
        std::cout << (char*)glGetString(GL_VERSION) << std::endl;
    }

    if (!scm::ogl.get().is_supported("GL_EXT_framebuffer_object")) {
        std::cout << "GL_EXT_framebuffer_object not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "GL_EXT_framebuffer_object supported" << std::endl;
    }

    if (!scm::ogl.get().is_supported("GL_ARB_texture_rectangle")) {
        std::cout << "GL_ARB_texture_rectangle not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "GL_ARB_texture_rectangle supported" << std::endl;
    }
    if (!scm::gl::time_query::is_supported()) {
        std::cout << "scm::gl::time_query not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "scm::gl::time_query available" << std::endl;
    }

    colors.push_back(math::vec3f_t(.3f));
    colors.push_back(math::vec3f_t(.0f, .6f, .0f));
    colors.push_back(math::vec3f_t(.8f, .6f, .0f));
    colors.push_back(math::vec3f_t(.8f, .0f, .6f));
    colors.push_back(math::vec3f_t(.3f, .6f, .8f));
    colors.push_back(math::vec3f_t(.3f, .6f, .3f));
    colors.push_back(math::vec3f_t(.3f, .6f, .0f));
    colors.push_back(math::vec3f_t(.6f, .3f, .0f));
    colors.push_back(math::vec3f_t(.0f, .3f, .0f));
    colors.push_back(math::vec3f_t(.6f, .2f, .3f));
    colors.push_back(math::vec3f_t(.6f, .0f, .3f));
    colors.push_back(math::vec3f_t(.8f));
    colors.push_back(math::vec3f_t(.0f, .0f, .6f));
    colors.push_back(math::vec3f_t(.6f, .0f, .0f));
    colors.push_back(math::vec3f_t(.6f));
    colors.push_back(math::vec3f_t(.0f, .0f, .3f));
    colors.push_back(math::vec3f_t(.3f, .0f, .0f));
    colors.push_back(math::vec3f_t(.6f, .3f, .6f));

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    //glClearColor(0.2f,0.2f,0.2f,1);
    //glClearColor(0.0f,0.0f,0.0f,1);
    glClearColor(1.0f,1.0f,1.0f,1.0f);

    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    glClearStencil(0);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(2);

    if (!init_font_rendering()) {
        std::cout << "unable to create font texture" << std::endl;
        return (false);
    }
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

    _volrend_cross_planes.reset(new gl::volume_renderer_crossplanes_glsl());

    if (!_volrend_cross_planes->initialize()) {
        std::cout << "unable to initialize crossplane volume renderer" << std::endl;
        return (false);
    }

    _volrend_raycast.reset(new gl::volume_renderer_raycast_glsl());

    if (!_volrend_raycast->initialize()) {
        std::cout << "unable to initialize raycasting volume renderer" << std::endl;
        return (false);
    }

    _shader_program.reset(new scm::gl::program_object());
    _vertex_shader.reset(new scm::gl::shader_object(GL_VERTEX_SHADER));
    _fragment_shader.reset(new scm::gl::shader_object(GL_FRAGMENT_SHADER));

    // load shader code from files
    if (!_vertex_shader->set_source_code_from_file("./../../../src/app_basic_volume_renderer/shader/two_sided_vert.glsl")) {
        std::cout << "Error loadong vertex shader:" << std::endl;
        return (false);
    }
    if (!_fragment_shader->set_source_code_from_file("./../../../src/app_basic_volume_renderer/shader/two_sided_frag.glsl")) {
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


    float dif[4]    = {1.0, 1.0, 1.0, 1};
    float spc[4]    = {0.7, 0.7, 0.9, 1};
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

    con_mode_changed();
    _volrend_params._point_of_interest  = math::vec3f_t(.5f, .5f, .5f);
    _volrend_params._extend             = math::vec3f_t(1.f, 1.f, 1.f);
#if 1
    if (!open_geometry_file("E:/_devel/data/wfarm/wells_inactive_only.sgeom")) {
        return (false);
    }
    if (!open_geometry_file("E:/_devel/data/wfarm/wells_active_only.sgeom")) {
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
    _volrend_raycast.reset();
    glDeleteFramebuffersEXT(1, &fbo_id);
    glDeleteTextures(1, &fbo_color_id);
    glDeleteTextures(1, &fbo_depth_id);

    scm::gl::shutdown();
    scm::shutdown();
}

void draw_console()
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

    if (_con_mode != console_hide) {
        _console_rend->draw();
    }


    // restore previous projection
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void fill_background()
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    // save polygon and depth buffer bit
    // to restore culling and depth mask settings later
    glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT);

    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);

    // disable depth writes
    glDepthMask(false);

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

        glBegin(GL_QUADS);
            glColor3f(0.26f, 0.3f, 0.4f);
            glVertex2f(0,0);
            glVertex2f(1,0);
            glColor3f(0.6f, 0.7f ,0.75f);
            glVertex2f(1,1);
            glVertex2f(0,1);
        glEnd();


    // restore the saved modelview matrix
    glPopMatrix();
    // restore the saved projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    // restore the saved polygon and depth buffer bits
    // to reset the culling and depth mask settings
    glPopAttrib();
    
    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}


void draw_texture_rect(unsigned tex_id,
                       const math::vec2ui_t tex_dim,
                       const math::vec2ui_t& ll,
                       const math::vec2ui_t& ur)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(ll.x, ll.y, ur.x, ur.y);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_id);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(  0.0f, 0.0f);
        glTexCoord2f(float(tex_dim.x), 0.0f);
        glVertex2f(  1.0f, 0.0f);
        glTexCoord2f(float(tex_dim.x), float(tex_dim.y));
        glVertex2f(  1.0f, 1.0f);
        glTexCoord2f(0.0f, float(tex_dim.y));
        glVertex2f(  0.0f, 1.0f);
    glEnd();


    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    glPopAttrib();

    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void draw_black_rect(const math::vec2ui_t tex_dim,
                     const math::vec2ui_t& ll,
                     const math::vec2ui_t& ur)
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(ll.x, ll.y, ur.x, ur.y);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();


    glDisable(GL_LIGHTING);

    glColor3f(0, 0, 0);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(  0.0f, 0.0f);
        glTexCoord2f(float(tex_dim.x), 0.0f);
        glVertex2f(  1.0f, 0.0f);
        glTexCoord2f(float(tex_dim.x), float(tex_dim.y));
        glVertex2f(  1.0f, 1.0f);
        glTexCoord2f(0.0f, float(tex_dim.y));
        glVertex2f(  0.0f, 1.0f);
    glEnd();



    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    glPopAttrib();

    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}


void display()
{
    static scm::time::high_res_timer    _timer;
    static scm::gl::time_query          _gl_timer;
    static double                       _accum_time     = 0.0;
    static double                       _gl_accum_time  = 0.0;
    static unsigned                     _accum_count    = 0;
    static scm::gl::axes_compass        compass;

    if (_high_quality_volume) {
        _volrend_params._step_size = 2048;
    }
    else {
        _volrend_params._step_size = 100;
    }

    _gl_timer.start();

    // clear the color and depth buffer
    if (draw_geometry) {
        glClear(GL_DEPTH_BUFFER_BIT | /*GL_COLOR_BUFFER_BIT | */ GL_STENCIL_BUFFER_BIT);
    }
    else {
        glClear(GL_DEPTH_BUFFER_BIT | /*GL_COLOR_BUFFER_BIT |*/ GL_STENCIL_BUFFER_BIT);
        //fill_background();
    }

    // push current modelview matrix
    glPushMatrix();

        // apply camera transform
        _trackball_manip.apply_transform();
        
        // geometry pass
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        //fill_background();
        render_geometry();
        //_volrend_raycast->draw_outlines(_volrend_params);

        glPushAttrib(GL_POLYGON_BIT | GL_COLOR_BUFFER_BIT);
        //glClear(GL_DEPTH_BUFFER_BIT );
            glFrontFace(GL_CCW);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);
            glColorMask(false, false, false, false);
            _volrend_raycast->draw_bounding_volume(_volrend_params);
        glPopAttrib();

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        math::vec2ui_t  _viewport_dim = math::vec2ui_t(winx, winy);

        if (draw_geometry) {
            draw_geometry_color_buffer();
        }

        if (   _show_image == 3
            || _show_image == 4
            || _show_image == 5) {
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

            if (_show_image != 3) {
                // volume pass
                render_volume();
            }
            else {
                //glClear(GL_COLOR_BUFFER_BIT);
                draw_black_rect(_viewport_dim, math::vec2ui_t(0, 0), _viewport_dim);
            }

            glPopAttrib();
        }
        else if (   _show_image == 0
                 || _show_image == 1
                 || _show_image == 2){
            // volume pass
            render_volume();
        }


        switch (_show_image) {
            //case 1:
            //case 5:draw_texture_rect(fbo_depth_id, _viewport_dim, vec2ui_t(0, 0), _viewport_dim);break;
            case 2:
            case 4:draw_texture_rect(fbo_depth_id, _viewport_dim, math::vec2ui_t(0, 0), _viewport_dim);break;
        }

    // restore previously saved modelview matrix
    //compass.render();
    glPopMatrix();
    //phong_shader->unbind();

    _gl_timer.stop();

    draw_console();
    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();

    _gl_timer.collect_result();

    _timer.stop();
    _timer.start();

    _volrend_params._last_frame_time = scm::time::to_milliseconds(_timer.get_time());

    _accum_time         += scm::time::to_milliseconds(_timer.get_time());
    _gl_accum_time      += scm::time::to_milliseconds(_gl_timer.get_time());
    ++_accum_count;

    if (_accum_time > 200.0) {
        std::stringstream   output;

        output.precision(2);
        output << std::fixed << "frame_time: " << _accum_time / static_cast<double>(_accum_count) << "msec "
                             << "gl time: " << _gl_accum_time / static_cast<double>(_accum_count) << "msec "
                             << "fps: " << static_cast<double>(_accum_count) / (_accum_time / 1000.0)
                             << std::endl;

        scm::console.get() << output.str();

        _accum_time     = 0.0;
        _gl_accum_time  = 0.0;
        _accum_count    = 0;
    }
}

void resize_n(int w, int h, float znear)
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
    gluPerspective(60.f, float(w)/float(h), znear, 5.f);


    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void resize(int w, int h)
{
    resize_n(w, h, _near_plane);
}

void keyboard(unsigned char key, int x, int y)
{
    int modifier = glutGetModifiers();

    int alt_pressed = modifier & GLUT_ACTIVE_ALT;

    switch (key) {
        // ESC key
        case 'a': _anim_step += 0.1f;break;
        case 'A': _anim_enabled = !_anim_enabled;break;
        case 'o':
        case 'O': open_volume(); break;
        case 'u':
        case 'U': open_unc_volume(); break;
        case 'g':
        case 'G': open_geometry();break;
        case 'q': _volrend_params._cp_pos.x += ui_float_increment * _volrend_params._aspect.x;break;
        case 'Q': _volrend_params._cp_pos.x -= ui_float_increment * _volrend_params._aspect.x;break;
        case 'w': _volrend_params._cp_pos.y += ui_float_increment * _volrend_params._aspect.y;break;
        case 'W': _volrend_params._cp_pos.y -= ui_float_increment * _volrend_params._aspect.y;break;
        case 'e': _volrend_params._cp_pos.z += ui_float_increment * _volrend_params._aspect.z;break;
        case 'E': _volrend_params._cp_pos.z -= ui_float_increment * _volrend_params._aspect.z;break;
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
        case '+': _near_plane += 0.01f; resize(winx, winy);break;
        case '-': _near_plane -= 0.01f; resize(winx, winy);break;
        case 's':
        case 'S': use_stencil_test = !use_stencil_test;break;
        //case 'i':
        //case 'I': do_inside_pass = !do_inside_pass;
        //          _volrend_raycast->do_inside_pass(do_inside_pass);
        //          break;
        case 'd':
        case 'D': draw_geometry = !draw_geometry;break;
        case 'p':
        case 'P':system("pause");break;
        case 'h':
        case 'H':_high_quality_volume = !_high_quality_volume;break;
        case 'c':
        case 'C':
            if (_con_mode == console_full)
                _con_mode = console_brief;
            else if (_con_mode == console_brief)
                _con_mode = console_hide;
            else if (_con_mode == console_hide)
                _con_mode = console_full;
            con_mode_changed();
            break;
        case 'i':
        case 'I': if (_show_image == 5) _show_image = 0;
                  else ++_show_image;
                  //if (   _show_image == 0
                  //    || _show_image == 2) {
                  //    resize_n(winx, winy, 0.01f);
                  //}
                  //else {
                  //    resize_n(winx, winy, 0.01f);
                  //}
                  break;
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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_STENCIL);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("OpenGL - basic volume renderer");

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


#include <scm/core/utilities/boost_warning_enable.h>
