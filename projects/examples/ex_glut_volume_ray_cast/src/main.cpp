
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/io/ios_state.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <scm/core.h>
#include <scm/log.h>
#include <scm/core/pointer_types.h>
#include <scm/core/io/tools.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_1d.h>
#include <scm/gl_util/data/analysis/transfer_function/build_lookup_table.h>
#include <scm/gl_util/data/volume/volume_reader_raw.h>
#include <scm/gl_util/data/volume/volume_reader_segy.h>
#include <scm/gl_util/data/volume/volume_reader_vgeo.h>

#include <scm/gl_core.h>
#include <scm/gl_core/buffer_objects/uniform_buffer_adaptor.h>

#include <scm/gl_util/data/imaging/texture_loader.h>
#include <scm/gl_util/data/volume/volume_loader.h>

#include <scm/gl_util/manipulators/trackball_manipulator.h>
#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/primitives/wavefront_obj.h>

//#include <GL/gl.h>
#include <GL/freeglut.h>
//#include <GL/gl3.h>

static bool fraps_bug = true;

static int winx = 600;
static int winy = 600;

const scm::math::vec3f diffuse(0.7f, 0.7f, 0.7f);
const scm::math::vec3f specular(0.2f, 0.7f, 0.9f);
const scm::math::vec3f ambient(0.1f, 0.1f, 0.1f);
const scm::math::vec3f position(1, 1, 1);

class demo_app
{
    struct transform_matrices
    {
        transform_matrices()
          : _mv_matrix(scm::math::mat4f::identity()),
            _mv_matrix_inverse(scm::math::mat4f::identity()),
            _mv_matrix_inverse_transpose(scm::math::mat4f::identity()),
            _p_matrix(scm::math::mat4f::identity()),
            _p_matrix_inverse(scm::math::mat4f::identity()),
            _mvp_matrix(scm::math::mat4f::identity()),
            _mvp_matrix_inverse(scm::math::mat4f::identity())
        {}

        scm::math::mat4f _mv_matrix;
        scm::math::mat4f _mv_matrix_inverse;
        scm::math::mat4f _mv_matrix_inverse_transpose;

        scm::math::mat4f _p_matrix;
        scm::math::mat4f _p_matrix_inverse;

        scm::math::mat4f _mvp_matrix;
        scm::math::mat4f _mvp_matrix_inverse;
    };

public:
    demo_app() {
        _initx = 0;
        _inity = 0;

        _lb_down = false;
        _mb_down = false;
        _rb_down = false;

        _dolly_sens = 10.0f;
        _sampling_distance = 1.0f;
    }
    virtual ~demo_app();

    bool initialize();
    void display();
    void resize(int w, int h);
    void mousefunc(int button, int state, int x, int y);
    void mousemotion(int x, int y);
    void keyboard(unsigned char key, int x, int y);

protected:
    scm::gl::texture_3d_ptr load_volume(scm::gl::render_device& in_device, const std::string& in_file_name) const;
    scm::gl::texture_1d_ptr create_color_map(scm::gl::render_device& in_device,
                                             unsigned in_size,
                                             const scm::data::piecewise_function_1d<float, float>& in_alpha,
                                             const scm::data::piecewise_function_1d<float, scm::math::vec3f>& in_color) const;

    void            update_transforms(const scm::gl::render_context& in_context);

private:
    scm::gl::trackball_manipulator _trackball_manip;
    float _initx;
    float _inity;

    bool _lb_down;
    bool _mb_down;
    bool _rb_down;

    float _dolly_sens;

    float               _sampling_distance;
    scm::math::vec3f    _max_volume_bounds;

    scm::data::piecewise_function_1d<float, float>                  _alpha_transfer;
    scm::data::piecewise_function_1d<float, scm::math::vec3f>       _color_transfer;

    scm::shared_ptr<scm::gl::render_device>     _device;
    scm::shared_ptr<scm::gl::render_context>    _context;

    scm::gl::program_ptr                _shader_program;

    scm::shared_ptr<scm::gl::box_geometry>  _box;

    scm::gl::texture_3d_ptr             _volume_texture;
    scm::gl::texture_1d_ptr             _colormap_texture;

    scm::gl::depth_stencil_state_ptr    _depth_less;

    scm::gl::blend_state_ptr            _no_blend;
    scm::gl::blend_state_ptr            _blend_omsa;

    scm::gl::sampler_state_ptr          _filter_linear;
    scm::gl::sampler_state_ptr          _filter_nearest;


    scm::gl::texture_2d_ptr             _color_buffer;
    scm::gl::texture_2d_ptr             _depth_buffer;
    scm::gl::frame_buffer_ptr           _framebuffer;
    scm::shared_ptr<scm::gl::quad_geometry>  _quad;
    scm::gl::program_ptr                _pass_through_shader;
    scm::gl::depth_stencil_state_ptr    _depth_no_z;
    scm::gl::rasterizer_state_ptr       _no_back_face_cull;

    scm::gl::rasterizer_state_ptr       _cull_back;

    scm::gl::program_ptr                _zfill_shader;
    scm::gl::frame_buffer_ptr           _zfill_framebuffer;
    scm::gl::rasterizer_state_ptr       _zfill_cull_front;

    scm::gl::timer_query_ptr            _timer_zfill;
    scm::gl::timer_query_ptr            _timer_volume;

    scm::gl::uniform_block<transform_matrices> _current_transforms;
    //scm::shared_ptr<transform_matrices> _current_transforms;
    //scm::gl::buffer_ptr                 _current_transforms_buffer;

}; // class demo_app


namespace  {

scm::scoped_ptr<demo_app> _application;

} // namespace

demo_app::~demo_app()
{
    _shader_program.reset();

    _box.reset();

    _volume_texture.reset();
    _colormap_texture.reset();

    _no_blend.reset();
    _blend_omsa.reset();

    _depth_less.reset();

    _filter_linear.reset();
    _filter_nearest.reset();

    _color_buffer.reset();
    _depth_buffer.reset();
    _framebuffer.reset();
    _quad.reset();
    _pass_through_shader.reset();
    _depth_no_z.reset();
    _no_back_face_cull.reset();

    _cull_back.reset();

    _zfill_shader.reset();
    _zfill_framebuffer.reset();
    _zfill_cull_front.reset();

    _timer_zfill.reset();
    _timer_volume.reset();

    _current_transforms.reset();

    _context.reset();
    _device.reset();
}

scm::gl::texture_3d_ptr
demo_app::load_volume(scm::gl::render_device& in_device, const std::string& in_file_name) const
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using namespace boost::filesystem;

    scoped_ptr<gl::volume_reader> vol_reader;
    path                    file_path(in_file_name);
    std::string             file_name       = file_path.filename().string();
    std::string             file_extension  = file_path.extension().string();

    data_format volume_data_format     = FORMAT_NULL;

    boost::algorithm::to_lower(file_extension);

    if (file_extension == ".raw") {
        vol_reader.reset(new scm::gl::volume_reader_raw(file_path.string(), false));
    }
    else if (file_extension == ".vol") {
        vol_reader.reset(new scm::gl::volume_reader_vgeo(file_path.string(), true));
    }
    else {
        std::cout << "demo_app::load_volume(): unsupported volume file format ('" << file_extension << "')." << std::endl;
        return (texture_3d_ptr());
    }

    if (!(*vol_reader)) {
        std::cout << "demo_app::load_volume(): unable to open file ('" << in_file_name << "')." << std::endl;
        return (texture_3d_ptr());
    }

    int    max_volume_dim  = in_device.capabilities()._max_texture_3d_size;
    vec3ui data_offset = vec3ui(0);
    vec3ui data_dimensions = vol_reader->dimensions();
    volume_data_format     = vol_reader->format();

    if (max(max(data_dimensions.x, data_dimensions.y), data_dimensions.z) > static_cast<unsigned>(max_volume_dim)) {
        std::cout << "demo_app::load_volume(): volume too large to load as single texture ('" << data_dimensions << "')." << std::endl;
        return (texture_3d_ptr());
    }

    scm::shared_array<unsigned char>    read_buffer;
    scm::size_t                         read_buffer_size =   data_dimensions.x * data_dimensions.y * data_dimensions.z
                                                           * size_of_format(volume_data_format);

    read_buffer.reset(new unsigned char[read_buffer_size]);

    if (!vol_reader->read(data_offset, data_dimensions, read_buffer.get())) {
        std::cout << "demo_app::load_volume(): unable to read data from file ('" << in_file_name << "')." << std::endl;
        return (texture_3d_ptr());
    }

    if (volume_data_format == FORMAT_NULL) {
        std::cout << "demo_app::load_volume(): unable to determine volume data format ('" << in_file_name << "')." << std::endl;
        return (texture_3d_ptr());
    }

    std::vector<void*> in_data;
    in_data.push_back(read_buffer.get());
    texture_3d_ptr new_volume_tex =
        in_device.create_texture_3d(data_dimensions, volume_data_format, 1, volume_data_format, in_data);

    return (new_volume_tex);
}

scm::gl::texture_1d_ptr
demo_app::create_color_map(scm::gl::render_device& in_device,
                           unsigned in_size,
                           const scm::data::piecewise_function_1d<float, float>& in_alpha,
                           const scm::data::piecewise_function_1d<float, scm::math::vec3f>& in_color) const
{
    using namespace scm::gl;
    using namespace scm::math;

    scm::scoped_array<scm::math::vec3f>  color_lut;
    scm::scoped_array<float>             alpha_lut;

    color_lut.reset(new vec3f[in_size]);
    alpha_lut.reset(new float[in_size]);

    if (   !scm::data::build_lookup_table(color_lut, in_color, in_size)
        || !scm::data::build_lookup_table(alpha_lut, in_alpha, in_size)) {
        std::cout << "demo_app::create_color_map(): error during lookuptable generation" << std::endl;
        return (texture_1d_ptr());
    }
    scm::scoped_array<float> combined_lut;

    combined_lut.reset(new float[in_size * 4]);

    for (unsigned i = 0; i < in_size; ++i) {
        combined_lut[i*4   ] = color_lut[i].x;
        combined_lut[i*4 +1] = color_lut[i].y;
        combined_lut[i*4 +2] = color_lut[i].z;
        combined_lut[i*4 +3] = alpha_lut[i];
    }

    std::vector<void*> in_data;
    in_data.push_back(combined_lut.get());

    texture_1d_ptr new_tex = in_device.create_texture_1d(in_size, FORMAT_RGBA_8, 1, 1, FORMAT_RGBA_32F, in_data);

    if (!new_tex) {
        std::cout << "demo_app::create_color_map(): error during color map texture generation." << std::endl;
        return (texture_1d_ptr());
    }

    return (new_tex);
}

bool
demo_app::initialize()
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;


    // initialize transfer functions //////////////////////////////////////////////////////////////
    _alpha_transfer.clear();
    _color_transfer.clear();

#if 0
    _alpha_transfer.add_stop(0,       1.0f);
    _alpha_transfer.add_stop(0.33f,   0.5f);
    _alpha_transfer.add_stop(0.40f,   0.0f);
    _alpha_transfer.add_stop(0.60f,   0.0f);
    _alpha_transfer.add_stop(0.66f,   0.5f);
    _alpha_transfer.add_stop(1.0f,    1.0f);
#else
    _alpha_transfer.add_stop(0,       0.0f);
    _alpha_transfer.add_stop(1.0f,    1.0f);
#endif

#if 0
    // blue-grey-orange
    _color_transfer.add_stop(0,       vec3f(0.0f, 1.0f, 1.0f));
    _color_transfer.add_stop(0.25f,   vec3f(0.0f, 0.0f, 1.0f));
    _color_transfer.add_stop(0.375f,  vec3f(0.256637f, 0.243243f, 0.245614f));
    _color_transfer.add_stop(0.50f,   vec3f(0.765487f, 0.738739f, 0.72807f));
    _color_transfer.add_stop(0.625f,  vec3f(0.530973f, 0.27027f, 0.0f));
    _color_transfer.add_stop(0.75f,   vec3f(1.0f, 0.333333f, 0.0f));
    _color_transfer.add_stop(1.0f,    vec3f(1.0f, 1.0f, 0.0f));
#else
    // blue-white-red
    _color_transfer.add_stop(0.0f, vec3f(0.0f, 0.0f, 1.0f));
    _color_transfer.add_stop(0.5f, vec3f(1.0f, 1.0f, 1.0f));
    _color_transfer.add_stop(1.0f, vec3f(1.0f, 0.0f, 0.0f));
#endif
    // initialize shaders /////////////////////////////////////////////////////////////////////////
    std::string vs_source;
    std::string fs_source;

    if (   !io::read_text_file("../../../src/shaders/volume_ray_cast.glslv", vs_source)
        || !io::read_text_file("../../../src/shaders/volume_ray_cast.glslf", fs_source)) {
        scm::err() << "error reading shader files" << log::end;
        return (false);
    }

    _device.reset(new scm::gl::render_device());
    _context = _device->main_context();
    scm::out() << *_device << scm::log::end;

    _shader_program = _device->create_program(list_of(_device->create_shader(STAGE_VERTEX_SHADER, vs_source))
                                                     (_device->create_shader(STAGE_FRAGMENT_SHADER, fs_source)));

    if (!_shader_program) {
        scm::err() << "error creating shader program" << log::end;
        return (false);
    }

    // initialize state objects ///////////////////////////////////////////////////////////////////
    _depth_less = _device->create_depth_stencil_state(true, true, COMPARISON_LESS);

    _no_blend   = _device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _blend_omsa = _device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);

    // initialize textures ////////////////////////////////////////////////////////////////////////
    _filter_linear  = _device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);
    _filter_nearest = _device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);

    volume_loader loader;
    
    _volume_texture = loader.load_texture_3d(*_device, "../../../res/volume/Engine_w256_h256_d256_c1_b8.raw", false);


    //_volume_texture = load_volume(*_device, "f:/data/src/Lux_Christopher_20999_4_1_mm_WT_w512_h512_d202_c1_b16.raw");
    //_volume_texture = load_volume(*_device, "../../../res/volume/Engine_w256_h256_d256_c1_b8.raw");
    //_volume_texture = load_volume(*_device, "e:/data/volume/vrgeo/new_zealand/pari_full_rm_8float_bri_TRIMMED.vol");
    //_volume_texture = load_volume(*_device, "d:/data/src/Engine_w256_h256_d256_c1_b8.raw");
    //_volume_texture = load_volume(*_device, "f:/data/src/wfarm_w512_h439_d512_c1_b8.raw");
    //_volume_texture = load_volume(*_device, "f:/data/src/mig_dep_0_2200.bri.vol");
    //_volume_texture = load_volume(*_device, "e:/data/gulfaks/gfaks.vol");
    //_volume_texture = load_volume(*_device, "e:/data/volume/general volumes/foot_w256_h256_d256_c1_b8.raw");
    _colormap_texture = create_color_map(*_device, 512, _alpha_transfer, _color_transfer);

    // initialize geometry ////////////////////////////////////////////////////////////////////////
    unsigned max_dim = max(max(_volume_texture->descriptor()._size.x,
                               _volume_texture->descriptor()._size.y),
                               _volume_texture->descriptor()._size.z);
    _max_volume_bounds = vec3f(_volume_texture->descriptor()._size) / max_dim;
    _sampling_distance = .5f / max_dim;

    _box = make_shared<box_geometry>(_device, vec3f(0.0f), _max_volume_bounds);//vec3f(-0.5f), vec3f(0.5f)));

    // initialize framebuffer /////////////////////////////////////////////////////////////////////
    texture_loader tex_loader;
    //_color_buffer =  tex_loader.load_texture_2d(*_device, "e:/working_copies/schism_x64/resources/textures/DH213SN.hdr", true, true);
    _color_buffer = _device->create_texture_2d(vec2ui(winx, winy), FORMAT_RGBA_16F);
    _depth_buffer = _device->create_texture_2d(vec2ui(winx, winy), FORMAT_D24);
    _framebuffer  = _device->create_frame_buffer();

    _framebuffer->attach_color_buffer(0, _color_buffer);
    _framebuffer->attach_depth_stencil_buffer(_depth_buffer);

    _quad.reset(new quad_geometry(_device, vec2f(0.0f, 0.0f), vec2f(1.0f, 1.0f)));
    _depth_no_z = _device->create_depth_stencil_state(false, false);
    _no_back_face_cull = _device->create_rasterizer_state(FILL_SOLID, CULL_NONE);

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
            out_color = texelFetch(in_texture, ivec2(gl_FragCoord.xy), 0).rgba;\
        }\
        ";
    ////texture(in_texture, tex_coord);
    _pass_through_shader = _device->create_program(list_of(_device->create_shader(STAGE_VERTEX_SHADER, v_pass))
                                                          (_device->create_shader(STAGE_FRAGMENT_SHADER, f_pass)));

    std::string v_zfill = "\
        #version 330\n\
        layout(std140, column_major) uniform;\
        \
        uniform transform_matrices\
        {\
            mat4 mv_matrix;\
            mat4 mv_matrix_inverse;\
            mat4 mv_matrix_inverse_transpose;\
            \
            mat4 p_matrix;\
            mat4 p_matrix_inverse;\
            \
            mat4 mvp_matrix;\
            mat4 mvp_matrix_inverse;\
        } current_transform;\
        out vec3 obj_position;\
        layout(location = 0) in vec3 in_position;\
        void main()\
        {\
            obj_position = in_position;\
            gl_Position = current_transform.mvp_matrix * vec4(in_position, 1.0);\
        }\
        ";
    std::string f_zfill = "\
        #version 330\n\
        \
        in vec3 obj_position;\
        layout(location = 0) out vec4 out_color;\
        void main()\
        {\
            out_color = vec4(obj_position, 1.0);\
        }\
        ";
    _zfill_shader = _device->create_program(list_of(_device->create_shader(STAGE_VERTEX_SHADER, v_zfill))
                                                   (_device->create_shader(STAGE_FRAGMENT_SHADER, f_zfill)));
    _zfill_framebuffer = _device->create_frame_buffer();
    _zfill_framebuffer->attach_depth_stencil_buffer(_depth_buffer);
    _zfill_cull_front = _device->create_rasterizer_state(FILL_SOLID, CULL_FRONT);

    _cull_back = _device->create_rasterizer_state(FILL_SOLID, CULL_BACK);

    _timer_zfill    = _device->create_timer_query();
    _timer_volume   = _device->create_timer_query();


    _current_transforms        = make_uniform_block<transform_matrices>(_device);
    //_current_transforms        = scm::make_shared<transform_matrices>();
    //_current_transforms_buffer = _device->create_buffer(BIND_UNIFORM_BUFFER, USAGE_DYNAMIC_DRAW, sizeof(transform_matrices));

    // set clear color, which is used to fill the background on glClear
    //glClearColor(0.2f,0.2f,0.2f,1);

    // setup depth testing

    // set polygonmode to fill front and back faces
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glShadeModel(GL_SMOOTH);

    _trackball_manip.dolly(2.5f);

    return (true);
}

unsigned plah             = 0;
double  time_zfill_accum  = 0.0;
double  time_volume_accum = 0.0;
unsigned accum_count      = 0;

void
demo_app::display()
{
    using namespace scm::gl;
    using namespace scm::math;

    // clear the color and depth buffer
    //glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    mat4f    view_matrix         = _trackball_manip.transform_matrix();
    mat4f    model_matrix        = mat4f::identity();
    translate(model_matrix, -(_max_volume_bounds * 0.5f));

    _current_transforms.begin_manipulation(_context);
    _current_transforms->_mv_matrix                    = view_matrix * model_matrix;
    _current_transforms->_mv_matrix_inverse            = inverse(_current_transforms->_mv_matrix);
    _current_transforms->_mv_matrix_inverse_transpose  = inverse(_current_transforms->_mv_matrix_inverse);
    _current_transforms->_mvp_matrix                   = _current_transforms->_p_matrix * _current_transforms->_mv_matrix;
    _current_transforms->_mvp_matrix_inverse           = inverse(_current_transforms->_mvp_matrix);
    _current_transforms.end_manipulation();

    vec3f   camera_location =  vec3f(_current_transforms->_mv_matrix_inverse.column(3))
                             / _current_transforms->_mv_matrix_inverse.column(3).w;

    //update_transforms(*_context);

    _context->clear_default_color_buffer(FRAMEBUFFER_BACK, vec4f(0.2f, .2f, .2f, 1.0f));
    _context->clear_default_depth_stencil_buffer();

    { // volume pass
        context_uniform_buffer_guard ubg(_context);

        _context->bind_uniform_buffer(_current_transforms.block_buffer(), 0);

        _context->begin_query(_timer_zfill);
        { // fill z of back faces
            context_state_objects_guard csg(_context);
            context_texture_units_guard tug(_context);

            //_context->clear_depth_stencil_buffer(_zfill_framebuffer);
            //_context->set_frame_buffer(_zfill_framebuffer);

            _context->clear_color_buffer(_framebuffer, 0, vec4f( .2f, .2f, .2f, 1.0f));
            _context->clear_depth_stencil_buffer(_framebuffer);
            _context->set_frame_buffer(_framebuffer);

            //_zfill_shader->uniform("mvp", _current_transforms->_mvp_matrix);
            _zfill_shader->uniform_buffer("transform_matrices", 0);

            _context->set_depth_stencil_state(_depth_less);
            _context->set_rasterizer_state(_zfill_cull_front);
            _context->set_blend_state(_no_blend);

            _context->reset_texture_units();

            _context->bind_program(_zfill_shader);

            _box->draw(_context, geometry::MODE_SOLID);
        }
        _context->end_query(_timer_zfill);

        //if (0)
        _context->begin_query(_timer_volume);
        {
            context_state_objects_guard     csg(_context);
            context_texture_units_guard     tug(_context);
            context_uniform_buffer_guard    ubg(_context);

            _shader_program->uniform("camera_location",             camera_location);
            _shader_program->uniform("sampling_distance",           _sampling_distance);
            _shader_program->uniform("max_bounds",                  _max_volume_bounds);
            //_shader_program->uniform("projection_matrix",           _current_transforms->_p_matrix);
            //_shader_program->uniform("model_view_matrix",           _current_transforms->_mv_matrix);
            //_shader_program->uniform("model_view_matrix_inverse",   _current_transforms->_mv_matrix_inverse);
            //_shader_program->uniform("mvp_inverse",                 _current_transforms->_mvp_matrix_inverse);
            //_shader_program->uniform("model_view_matrix_inverse_transpose", mv_inv_transpose);
            _shader_program->uniform_sampler("volume_texture",    0);
            _shader_program->uniform_sampler("color_map_texture", 1);
            _shader_program->uniform_sampler("depth_texture",     2);

            _shader_program->uniform_buffer("transform_matrices", 0);

            _context->reset_state_objects();

            _context->set_default_frame_buffer();
            //_context->set_depth_stencil_state(_depth_less);
            _context->set_blend_state(_blend_omsa);
            //_context->set_rasterizer_state(_cull_back);

            _context->bind_program(_shader_program);

            _context->reset_texture_units();
            _context->bind_texture(_volume_texture, _filter_linear, 0);
            _context->bind_texture(_colormap_texture, _filter_linear, 1);
            //_context->bind_texture(_depth_buffer, _filter_nearest, 2);
            _context->bind_texture(_color_buffer, _filter_nearest, 2);

            _box->draw(_context, geometry::MODE_SOLID);
        }
        _context->end_query(_timer_volume);

        _context->reset_uniform_buffers();
    }

    //_context->collect_query_results(_timer_zfill);
    //_context->collect_query_results(_timer_volume);

    time_zfill_accum  += scm::time::to_milliseconds(scm::time::nanosec(_timer_zfill->result()));
    time_volume_accum += scm::time::to_milliseconds(scm::time::nanosec(_timer_volume->result()));
    accum_count       += 1;

    if ((time_zfill_accum + time_volume_accum) > 1000.0) {
        boost::io::ios_all_saver ias(std::cout);
        std::cout << "zfill time:  " << std::fixed << time_zfill_accum / accum_count  << "msec, "
                  << "volume time: " << std::fixed << time_volume_accum / accum_count << "msec" << std::endl;
        time_zfill_accum  = 0.0;
        time_volume_accum = 0.0;
        accum_count       = 0;
    }

    if (0)
    { // fullscreen pass
        mat4f   pass_mvp = mat4f::identity();
        ortho_matrix(pass_mvp, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

        _pass_through_shader->uniform_sampler("in_texture", 0);
        _pass_through_shader->uniform("mvp", pass_mvp);

        _context->set_default_frame_buffer();
        _context->set_depth_stencil_state(_depth_no_z);
        _context->set_blend_state(_no_blend);
        _context->set_rasterizer_state(_cull_back);
        _context->bind_texture(_color_buffer, _filter_nearest, 0);
        _context->bind_program(_pass_through_shader);

        _quad->draw(_context);
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
    //glViewport(0, 0, w, h);

    scm::math::perspective_matrix(_current_transforms->_p_matrix, 60.f, float(w)/float(h), 0.1f, 100.0f);
    _current_transforms->_p_matrix_inverse = inverse(_current_transforms->_p_matrix);
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
    scm::shared_ptr<scm::core>      scm_core(new scm::core(argc, argv));
    _application.reset(new demo_app());

    // the stuff that has to be done
    glutInit(&argc, argv);
    glutInitContextVersion(4, 3);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    //glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);

    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA);
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
