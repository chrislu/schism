
#include <iostream>

#include <set>
#include <string>

#include <scm_core/math/math.h>
#include <boost/date_time.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <ogl/gl.h>
#include <GL/glut.h>

#include <ogl/manipulators/trackball_manipulator.h>
#include <ogl/shader_objects/program_object.h>
#include <ogl/shader_objects/shader_object.h>
#include <ogl/time/time_query.h>

#include <scm_core/time/high_res_timer.h>

#include <scm_core/core.h>

#include <image_handling/image_loader.h>

#include <ft2build.h>
#include FT_FREETYPE_H
#include <freetype/ftmodapi.h>



boost::scoped_ptr<gl::program_object>  _shader_program;
boost::scoped_ptr<gl::shader_object>   _vertex_shader;
boost::scoped_ptr<gl::shader_object>   _fragment_shader;

gl::trackball_manipulator _trackball_manip;

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

unsigned tex_font = 0;
unsigned tex_font_w = 0;
unsigned tex_font_h = 0;

struct font_face
{
    math::vec2i_t       _tex_lower_left;
    math::vec2i_t       _tex_upper_right;

    unsigned            _hor_advance;

    math::vec2i_t       _bearing;
}; // struct font_face

typedef std::map<unsigned, font_face>      font_face_container;

font_face_container font_faces;

char font_face_kerning_table[256][256];

bool init_font_rendering()
{
    // soon to be parameters
    std::string     _font_file_name         = std::string("./../../../res/fonts/consolai.ttf");//vgaoem.fon");//cour.ttf");//
    unsigned        _font_size              = 18;
    unsigned        _display_dpi_resolution = 72;

    scm::time::high_res_timer _timer;
    _timer.start();

    FT_Library      ft_lib_;

    // initialize freetype
    if (FT_Init_FreeType(&ft_lib_) != 0) {
        std::cout << "error initializing freetype library" << std::endl;
        return (false);
    }

    const     FT_Library      ft_lib = ft_lib_;


    //switch (FT_Get_TrueType_Engine_Type(ft_lib)) {
    //    case FT_TRUETYPE_ENGINE_TYPE_NONE:          std::cout << "FT_TRUETYPE_ENGINE_TYPE_NONE"       << std::endl; break;
    //    case FT_TRUETYPE_ENGINE_TYPE_UNPATENTED:    std::cout << "FT_TRUETYPE_ENGINE_TYPE_UNPATENTED" << std::endl; break;
    //    case FT_TRUETYPE_ENGINE_TYPE_PATENTED:      std::cout << "FT_TRUETYPE_ENGINE_TYPE_PATENTED"   << std::endl; break;
    //}

    // load font face
    FT_Face         ft_face;
    if (FT_New_Face(ft_lib, _font_file_name.c_str(), 0, &ft_face) != 0) {
        std::cout << "error loading font file (" << _font_file_name << ")" << std::endl;
        return (false);
    }

    if (ft_face->face_flags & FT_FACE_FLAG_SCALABLE) {
        std::cout << "scalable font" << std::endl;
    }
    else if (ft_face->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        std::set<int> available_sizes;

        std::cout << "fixed size font" << std::endl;
        std::cout << "available sizes:" << std::endl;

        for (int i = 0; i < ft_face->num_fixed_sizes; ++i) {
            available_sizes.insert(ft_face->available_sizes[i].height);
        }

        if (available_sizes.empty()) {
            std::cout << "no sized defined" << std::endl;
            return (false);
        }

        // scale size to our current display resolution
        _font_size = (_display_dpi_resolution * _font_size + 36) / 72;

        // now find closest matching size
        std::set<int>::const_iterator lower_bound = available_sizes.lower_bound(_font_size); // first >=
        std::set<int>::const_iterator upper_bound = available_sizes.upper_bound(_font_size); // first >

        if (   upper_bound == available_sizes.end()) {
            _font_size = *available_sizes.rbegin();
        }
        else {
            _font_size = *lower_bound;
        }

        // ok bitmap fonts are in pixel sizes (i.e. 72dpi), so scale the size
        _font_size = (72 * _font_size + _display_dpi_resolution / 2) / _display_dpi_resolution;
    }
    else {
        std::cout << "unsupported font format" << std::endl;
    }

    // rendering the font bullshit into a texture
    if (FT_Set_Char_Size(ft_face, 0, _font_size << 6, 0, _display_dpi_resolution) != 0) {
        std::cout << "error setting font sizes" << std::endl;
        return (false);
    }

    math::vec2f_t bbox_x;
    math::vec2f_t bbox_y;

    if (ft_face->face_flags & FT_FACE_FLAG_SCALABLE) {
        float   em_size = 1.0 * ft_face->units_per_EM;
        float   x_scale =  ft_face->size->metrics.x_ppem / em_size;
        float   y_scale =  ft_face->size->metrics.y_ppem / em_size;

        bbox_x = math::vec2f_t(ft_face->bbox.xMin * x_scale,
                               ft_face->bbox.xMax * x_scale);
        bbox_y = math::vec2f_t(ft_face->bbox.yMin * y_scale,
                               ft_face->bbox.yMax * y_scale);
    }
    else if (ft_face->face_flags & FT_FACE_FLAG_FIXED_SIZES) {

        bbox_x = math::vec2f_t(0,
                               ft_face->size->metrics.max_advance >> 6);
        bbox_y = math::vec2f_t(0,
                               ft_face->size->metrics.height >> 6);
    }
  ///* convert design distances to floating point pixels */
  //std::cout << "xMin" << (ft_face->bbox.xMin *  x_scale ) << std::endl;
  //std::cout << "xMax" << (ft_face->bbox.xMax *  x_scale ) << std::endl;
  //std::cout << "yMin" << (ft_face->bbox.yMin *  y_scale ) << std::endl;
  //std::cout << "yMax" << (ft_face->bbox.yMax *  y_scale ) << std::endl;
  //std::cout << "aMax" << ((ft_face->size->metrics.max_advance >> 6)) << std::endl;
  //std::cout << "height" << (ft_face->height *  y_scale ) << std::endl;

  math::vec2i_t font_glyph_size  = math::vec2i_t(math::ceil(bbox_x.y) - math::floor(bbox_x.x),
                                                 math::ceil(bbox_y.y) - math::floor(bbox_y.x));

  std::cout << "bbox.x" << font_glyph_size.x << std::endl;
  std::cout << "bbox.y" << font_glyph_size.y << std::endl;
/////////////////////////
    //std::cout << "ma: " << (ft_face->size->metrics.max_advance & 63);
    //std::cout << "mh: " << (ft_face->size->metrics.height & 63);

    math::vec2i_t font_tex_size    = math::vec2i_t(font_glyph_size * 16);

    // allocate texture destination memory
    boost::scoped_array<unsigned char> font_tex(new unsigned char[font_tex_size.x * font_tex_size.y]);

    // clear texture background to black
    memset(font_tex.get(), 0u, font_tex_size.x * font_tex_size.y);

    unsigned dst_x, dst_y;

    font_face cur_font_face;

    for (unsigned i = 0; i < 256; ++i) {
        if(FT_Load_Glyph(ft_face, FT_Get_Char_Index(ft_face, i), FT_LOAD_DEFAULT)) {
            std::cout << "error loading glyph at char code: " << i << std::endl;
        }
        else {
            if (FT_Render_Glyph(ft_face->glyph, FT_RENDER_MODE_NORMAL)) {
                std::cout << "error rendering glyph at char code: " << i << std::endl;
            }
            FT_Bitmap& bitmap = ft_face->glyph->bitmap;

            //std::cout << i << " bx: " <<  (ft_face->glyph->metrics.horiBearingX >> 6)
            //               << "\tby: " << ((ft_face->glyph->metrics.height >> 6) - (ft_face->glyph->metrics.horiBearingY >> 6))
            //               << "\ta: " <<  (ft_face->glyph->metrics.horiAdvance  >> 6)<< std::endl;

            dst_x =                   (i & 0x0F) * font_glyph_size.x;
            dst_y = font_tex_size.y - ((i >> 4) + 1)   * font_glyph_size.y;

            math::vec2i_t ftex_size(0, bitmap.rows);

            switch (bitmap.pixel_mode) {
                case FT_PIXEL_MODE_GRAY:

                    ftex_size.x = bitmap.width;

                    //if (bitmap.pitch > font_glyph_size.x) {
                    //    std::cout << i << " " << (bitmap.pitch - font_glyph_size.x) << std::endl;
                    //}

                    for (int dy = 0; dy < bitmap.rows; ++dy) {

                        unsigned src_off = dy * bitmap.pitch;
                        unsigned dst_off = dst_x + (dst_y + bitmap.rows - 1 - dy) * font_tex_size.x;// + (font_glyph_size.y - bitmap.rows) * font_tex_size.x;
                        memcpy(font_tex.get() + dst_off, bitmap.buffer + src_off, bitmap.width);
                    }
                    break;
                case FT_PIXEL_MODE_MONO:
                    //std::cout << "r: " << bitmap.rows << "\tw: " << bitmap.width << "\tp: " << bitmap.pitch << std::endl;
                    ftex_size.x = bitmap.width;
                    for (int dy = 0; dy < bitmap.rows; ++dy) {
                        for (int dx = 0; dx < bitmap.pitch; ++dx) {

                            unsigned        src_off     = dx + dy * bitmap.pitch;
                            unsigned char   src_byte    = bitmap.buffer[src_off];

                            for (int bx = 0; bx < 8; ++bx) {

                                unsigned dst_off    = (dst_x + dx * 8 + bx) + (dst_y + bitmap.rows - 1 - dy) * font_tex_size.x;

                                unsigned char  src_set = src_byte & (0x80 >> bx);
                                unsigned char* plah = &src_byte;

                                font_tex[dst_off] = src_set ? 255u : 0u;
                            }
                        }
                    }
                    break;
                default:
                    std::cout << "unsupported pixel_mode" << std::endl;
                    continue;

            }
            cur_font_face._tex_lower_left   = math::vec2i_t(dst_x, dst_y);
            cur_font_face._tex_upper_right  = cur_font_face._tex_lower_left + ftex_size;

            cur_font_face._hor_advance      = ft_face->glyph->metrics.horiAdvance >> 6;
            cur_font_face._bearing          = math::vec2i_t(ft_face->glyph->metrics.horiBearingX >> 6,
                                                            (ft_face->glyph->metrics.horiBearingY >> 6) - (ft_face->glyph->metrics.height >> 6));

            font_faces.insert(font_face_container::value_type(i, cur_font_face));

            //std::cout << i << " w: " << bitmap.width << "\th: " << bitmap.rows << "\tp: " << bitmap.pitch << std::endl;

            //for (int dy = 0; dy < bitmap.rows; ++dy) {

            //    unsigned src_off = dy * bitmap.width;
            //    unsigned dst_off = dst_x + (dst_y + bitmap.rows - 1 - dy) * font_tex_size.x;
            //    memcpy(font_tex.get() + dst_off, bitmap.buffer + src_off, bitmap.width);
            //}



    //        //std::cout << i << " w: " << bitmap.width << "\th: " << bitmap.rows << "\tax: " << (ft_face->glyph->advance.x >> 6) << "\tay: " << 
    //                           //  (/*float(ft_face->max_advance_height) / float(ft_face->units_per_EM) * */ft_face->size->metrics.height >> 6) << std::endl;//(ft_face->glyph->linearVertAdvance >> 16) << "." << (ft_face->glyph->linearVertAdvance & 0x0000FFFF)<< std::endl;
    //        //std::cout << i << " w: " << (float)(ft_face->bbox.xMax - ft_face->bbox.xMin) / (float)64 << " h: " << (float)(ft_face->bbox.yMax - ft_face->bbox.yMin) / (float)64 << std::endl;


        }
    }


    if (ft_face->face_flags & FT_FACE_FLAG_KERNING) {
        std::cout << "has kerning" << std::endl;
        for (unsigned l = 0; l < 256; ++l) {
            FT_UInt l_glyph_index = FT_Get_Char_Index(ft_face, l);
            for (unsigned r = 0; r < 256; ++r) {
                FT_UInt r_glyph_index = FT_Get_Char_Index(ft_face, r);
                FT_Vector  delta;
                FT_Get_Kerning(ft_face, l_glyph_index, r_glyph_index,
                               FT_KERNING_DEFAULT, &delta);

                //if ((delta.x >> 6) != 0) {
                //    std::cout << l << " " << r << (delta.x >> 6) << std::endl;
                //}

                font_face_kerning_table[l][r] = (delta.x >> 6);

            }
        }
    }
    else {
        for (unsigned l = 0; l < 256; ++l) {
            for (unsigned r = 0; r < 256; ++r) {

                font_face_kerning_table[l][r] = 0;
            }
        }
    }





    // shutdown font face
    if (FT_Done_Face(ft_face) != 0) {
        std::cout << "error closing font face" << std::endl;
        return (false);
    }

    // shutfdown freetype
    if (FT_Done_FreeType(ft_lib) != 0) {
        std::cout << "error shutting down freetype library" << std::endl;
    }


    glGenTextures(1, &tex_font);

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_font);
    glEnable(GL_TEXTURE_RECTANGLE_ARB);


    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_ALPHA, font_tex_size.x, font_tex_size.y, 0, GL_ALPHA, GL_UNSIGNED_BYTE, (void*)font_tex.get());
    if (glGetError() != GL_NO_ERROR) {
        std::cout << "error: texture upload failed" << std::endl;
        return (false);
    }
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    tex_font_w = font_tex_size.x;
    tex_font_h = font_tex_size.y;

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
    if (!gl::is_supported("GL_VERSION_2_0")) {
        std::cout << "GL_VERSION_2_0 not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "OpenGL 2.0 supported, OpenGL Version: " << (char*)glGetString(GL_VERSION) << std::endl;
    }
    if (!gl::time_query::is_supported()) {
        std::cout << "gl::time_query not supported" << std::endl;
        return (false);
    }
    else {
        std::cout << "gl::time_query available" << std::endl;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);

    // set clear color, which is used to fill the background on glClear
    glClearColor(0.2f,0.2f,0.2f,1);
    //glClearColor(0.8f,0.8f,0.8f,1);

    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    _trackball_manip.dolly(1);

    _shader_program.reset(new gl::program_object());
    _vertex_shader.reset(new gl::shader_object(GL_VERTEX_SHADER));
    _fragment_shader.reset(new gl::shader_object(GL_FRAGMENT_SHADER));

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

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winx, 0, winy, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_font);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glTranslatef(20, 20, 0);

//glTranslatef(30, 30, 0);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(  0.0f, 0.0f);
        glTexCoord2f(tex_font_w, 0.0f);
        glVertex2f(  tex_font_w, 0.0f);
        glTexCoord2f(tex_font_w, tex_font_h);
        glVertex2f(  tex_font_w, tex_font_h);
        glTexCoord2f(0.0f, tex_font_h);
        glVertex2f(  0.0f, tex_font_h);
    glEnd();


    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glDisable(GL_BLEND);


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

void draw_string(const std::string& stuff_that_matters)
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winx, 0, winy, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    //glBlendFuncSeparate(GL_SRC_COLOR,
    //                    GL_ONE_MINUS_SRC_COLOR,
    //                    GL_SRC_COLOR,
    //                    GL_ONE_MINUS_SRC_COLOR);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_font);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    math::vec2i_t current_pos = math::vec2i_t(20 + 1, winy / 2 - 1);

    glColor4f(0, 0, 0, 1);

    unsigned prev = 0;

    for (std::string::const_iterator c = stuff_that_matters.begin();
         c != stuff_that_matters.end();
         ++c) {
        font_face cf = font_faces[*c];

        if (prev) {
            current_pos.x += font_face_kerning_table[prev][*c];
            //if (font_face_kerning_table[prev][*c]!= 0)
            //    std::cout << *c << font_face_kerning_table[prev][*c] << std::endl;
        }

        draw_quad(current_pos + cf._bearing,
                  current_pos + cf._bearing + cf._tex_upper_right - cf._tex_lower_left,
                  cf._tex_lower_left,
                  cf._tex_upper_right);

        current_pos.x += cf._hor_advance;
        prev = *c;
    }
    
    glColor4f(1, 1, 1, 1);
    current_pos = math::vec2i_t(20, winy / 2);
    //current_pos = math::vec2i_t(20 + 1, winy / 2 - 1 - 25);
    
    prev = 0;

    for (std::string::const_iterator c = stuff_that_matters.begin();
         c != stuff_that_matters.end();
         ++c) {
        font_face cf = font_faces[*c];

        if (prev) {
            current_pos.x += font_face_kerning_table[prev][*c];
        }

        draw_quad(current_pos + cf._bearing,
                  current_pos + cf._bearing + cf._tex_upper_right - cf._tex_lower_left,
                  cf._tex_lower_left,
                  cf._tex_upper_right);

        current_pos.x += cf._hor_advance;
        prev = *c;
    }


    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);


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
    static gl::time_query            _gl_timer;

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
#if 0
#endif
    //draw_font_image();

    draw_string("mjM hallo Welt! To ro  \n:\\  .\\ []{}|||~!@#$%^&*()_+;:<>/|'\"~` 0123456789");

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

//#include <boost/date_time/posix_time/posix_time.hpp>
//#include <boost/date_time/gregorian/gregorian.hpp>
//
//#include <scm_core/time/time_system.h>


#include <boost/shared_ptr.hpp>

int main(int argc, char **argv)
{
    struct plah
    {
        boost::weak_ptr<float>     _q;

        float& get() const {
            return (*_q.lock());
        }

    };

    plah pl;

    boost::shared_ptr<float>  p(new float);
    boost::shared_ptr<float>  p2(new float);

    *p  = 5.0f;
    *p2 = 4.0f;
    pl._q = boost::weak_ptr<float>(p);

    boost::weak_ptr<float>     q(p);
    boost::weak_ptr<float>     r(p);

    boost::weak_ptr<float>     r2(p2);

    *r.lock() = 6.0f;

    float& g = pl.get();

    float  f = pl.get();

    if (!(p < q.lock()) && !(q.lock() < p)) {//p == q.lock()) {
        std::cout << 3 << std::endl;
    }

    if (!(q < r) && !(r < q)) {
        std::cout << 1 << std::endl;
    }

    if (boost::shared_ptr<float> m = q.lock()) {
        std::cout << 2 << std::endl;
    }

    p.reset(new float);
    *p = 1.1f;

    r = r2;

    if (!(q < r) && !(r < q) ) {
        std::cout << 1 << std::endl;
    }

   // using namespace boost;

   // posix_time::ptime   t1(gregorian::day_clock::universal_day(), posix_time::hours(23) + posix_time::minutes(59) + posix_time::seconds(59)); //
   // posix_time::ptime   t2(gregorian::day_clock::universal_day(), posix_time::hours(0) + posix_time::minutes(0) + posix_time::seconds(1)); //
   // //posix_time::ptime   t1(gregorian::date(2002, 1, 1), posix_time::hours(23) + posix_time::minutes(59) + posix_time::seconds(59)); //
   // //posix_time::ptime   t2(gregorian::date(2002, 1, 2), posix_time::hours(0) + posix_time::minutes(0) + posix_time::seconds(1)); //
   //// posix_time::ptime   t2(gregorian::date(2002, 1, 1), scm::time::time_system::time_traits::nanosec(1000));//posix_time::nanosec(1)); //

   // std::cout << t1 << std::endl;

   // posix_time::time_duration d1 = t2 - t1;

   // std::cout << d1.is_special() << std::endl;
   // std::cout << d1 << std::endl;

    //std::cout << sizeof(boost::posix_time::ptime) << std::endl;

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



#if 0
    scm::time::timer    _timer;
    scm::time::timer    _timer2;
    system("pause");


    unsigned            _test_loop_count = 1000;

    // perfcounter
    for (unsigned k = 0; k < 10u; ++k) {
        _timer.start();
        for (unsigned i = 0; i < (_test_loop_count/* * pow(10.0f, (float)k)*/); ++i) {
            //_timer2.start();
            scm::core::detail::get_time();
        }
        _timer.stop();

        std::cout.precision(6);
        std::cout << "time for : " << std::fixed << _timer.get_time() / ((double)_test_loop_count /** pow(10.0f, (float)k)*/) << "usec" << std::endl << std::flush;
    }

    //_timer.start();
    //while ()) {
    //    scm::core::detail::get_time();
    //}
    //_timer.stop();

    //std::cout.precision(6);
    //std::cout << "time for : " << std::fixed << _timer.get_time() / (double)_test_loop_count << "msec" << std::endl;
    // ftime
    _timer.start();
    for (unsigned i = 0; i < _test_loop_count; ++i) {
        boost::posix_time::microsec_clock::universal_time();
    }

    _timer.stop();
    std::cout.precision(6);
    std::cout << "time for : " << std::fixed << _timer.get_time() / (double)_test_loop_count << "msec" << std::endl;

    ttrait::time_t start;
    ttrait::time_t end;

    start = ttrait::current_time();
    for (unsigned i = 0; i < _test_loop_count; ++i) {
        //_timer2.start();
        scm::core::detail::get_time();
    }
    end = ttrait::current_time();
    std::cout.precision(6);
    std::cout << "time for : " << std::fixed << ttrait::to_milliseconds(ttrait::difference(end, start)) / (double)_test_loop_count << "msec" << std::endl;

    start = ttrait::current_time();
    for (unsigned i = 0; i < _test_loop_count; ++i) {
        boost::posix_time::microsec_clock::universal_time();
    }
    end = ttrait::current_time();
    std::cout.precision(6);
    std::cout << "time for : " << std::fixed << ttrait::to_milliseconds(ttrait::difference(end, start)) / (double)_test_loop_count << "msec" << std::endl;
#endif

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


#if 0
    //scm::root.get().get_std_console_listener().set_log_threshold(scm::con::panic);
    //scm::console.get() << scm::con::log_level(scm::con::output)
    //                   << scm::root.get().get_version_string() << std::endl;
    //scm::console.get() << scm::con::log_level(scm::con::output)
    //                   << "sizeof(long) = " << sizeof(long) << std::endl;
    //Must match with time_resolutions enum in date_time/time_defs.h
    const char* const resolution_names[] = {"Second", "Deci", "Centi", "Milli",
    "Ten_Thousanth", "Micro", "Nano"};

    using namespace boost::posix_time;

    scm::console.get() << scm::con::log_level(scm::con::output)
                       << "Resolution: "
                       << resolution_names[time_duration::rep_type::resolution()] 
                       << " -- Ticks per second: "
                       << time_duration::rep_type::res_adjust() << std::endl;

#endif
#if 0
    scm::console.get() << "sick, sad world!" << std::endl /*<< std::endl*/
                       << std::fixed << 1.343434 << 666 << std::endl;

    scm::console.get() << scm::con::log_level(scm::con::warning) << "warning" << std::endl;
    scm::console.get() << scm::con::log_level(scm::con::debug) << "debug" << std::endl;
    scm::console.get() << scm::con::log_level(scm::con::error) << "error" << std::endl;
    scm::console.get() << scm::con::log_level(scm::con::panic) << "panic" << std::endl;

    scm::root.get().get_std_console_listener().set_log_threshold(scm::con::debug);

    scm::console.get() << scm::con::log_level(scm::con::warning) << "warning" << std::endl;
    scm::console.get() << scm::con::log_level(scm::con::debug) << "debug" << std::endl;
    scm::console.get() << scm::con::log_level(scm::con::error) << "error" << std::endl;
    scm::console.get() << scm::con::log_level(scm::con::panic) << "panic" << std::endl;
#endif

    // init the GL context
    if (!gl::initialize()) {
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