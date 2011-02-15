
#include "texture_output.h"

#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/primitives/quad.h>

namespace {

std::string fs_vsrc = "\
    #version 330\n\
    \n\
    uniform mat4 mvp;\n\
    out vec2 tex_coord;\n\
    \n\
    layout(location = 0) in vec3 in_position;\n\
    layout(location = 2) in vec2 in_texture_coord;\n\
    \n\
    void main() {\n\
        gl_Position = mvp * vec4(in_position, 1.0);\n\
        tex_coord   = in_texture_coord;\n\
    }\n\
    ";

std::string fs_color_fsrc = "\
    #version 330\n\
    \n\
    in vec2 tex_coord;\n\
    \n\
    uniform sampler2D in_texture;\n\
    \n\
    layout(location = 0) out vec4 out_color;\n\
    \n\
    void main() {\n\
        out_color = texture(in_texture, tex_coord).rgba;\n\
    }\n\
    ";

std::string fs_gray_fsrc = "\
    #version 330\n\
    \n\
    in vec2 tex_coord;\n\
    \n\
    uniform sampler2D in_texture;\n\
    \n\
    layout(location = 0) out vec4 out_color;\n\
    \n\
    void main() {\n\
        float f   = texture(in_texture, tex_coord).r;\n\
        out_color = vec4(vec3(f), 1.0);\n\
    }\n\
    ";

} // namespace

namespace scm {
namespace gl {

texture_output::texture_output(const gl::render_device_ptr& device)
{
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    mat4f pass_mvp = mat4f::identity();
    ortho_matrix(pass_mvp, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

    // for fs-quad
    _quad_geom = make_shared<quad_geometry>(device, vec2f(0.0f, 0.0f), vec2f(1.0f, 1.0f));

    _fs_program_color = device->create_program(list_of(device->create_shader(scm::gl::STAGE_VERTEX_SHADER,   fs_vsrc,       "texture_output::fs_vsrc"))
                                                      (device->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, fs_color_fsrc, "texture_output::fs_color_fsrc")),
                                               "texture_output::fs_program_color");

    if(!_fs_program_color) {
        throw (std::runtime_error("texture_output::texture_output(): error generating _fs_program_color program"));
    }
    _fs_program_color->uniform("in_texture", 0);
    _fs_program_color->uniform("mvp", pass_mvp);

    _fs_program_gray = device->create_program(list_of(device->create_shader(scm::gl::STAGE_VERTEX_SHADER,   fs_vsrc,      "texture_output::fs_vsrc"))
                                                     (device->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, fs_gray_fsrc, "texture_output::fs_gray_fsrc")),
                                               "texture_output::fs_program_gray");

    if(!_fs_program_gray) {
        throw (std::runtime_error("texture_output::texture_output(): error generating _fs_program_gray program"));
    }
    _fs_program_gray->uniform("in_texture", 0);
    _fs_program_gray->uniform("mvp", pass_mvp);

    _filter_nearest    = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _dstate_no_z_write = device->create_depth_stencil_state(false, false);
    _rstate_cull_back  = device->create_rasterizer_state(FILL_SOLID, CULL_BACK);
    _bstate_no_blend   = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
}

texture_output::~texture_output()
{
    _quad_geom.reset();
    _fs_program_color.reset();
    _fs_program_gray.reset();

    _filter_nearest.reset();
    _dstate_no_z_write.reset();
    _rstate_cull_back.reset();
    _bstate_no_blend.reset();
}

void
texture_output::draw_texture_2d(const gl::render_context_ptr& context,
                                const gl::texture_2d_ptr&     tex,
                                const math::vec2ui&           position,
                                const math::vec2ui&           extend) const
{
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_texture_units_guard tug(context);
    context_framebuffer_guard   fbg(context);

    vec2ui tile_size = vec2ui(0);

    float aspect =  static_cast<float>(tex->descriptor()._size.x)
                  / static_cast<float>(tex->descriptor()._size.y);

    tile_size.x = extend.x;
    tile_size.y = static_cast<unsigned>(tile_size.x / aspect);

    if (tile_size.y > extend.y) {
        tile_size.y = extend.y;
        tile_size.x = static_cast<unsigned>(tile_size.y * aspect);
    }

    context->set_depth_stencil_state(_dstate_no_z_write);
    context->set_blend_state(_bstate_no_blend);
    context->set_rasterizer_state(_rstate_cull_back);

    context->set_viewport(viewport(position, tile_size));

    if (channel_count(tex->descriptor()._format) > 1) {
        context->bind_program(_fs_program_color);
    }
    else {
        context->bind_program(_fs_program_gray);
    }
    context->bind_texture(tex, _filter_nearest, 0/*texture unit 0*/);

    _quad_geom->draw(context, geometry::MODE_SOLID);
}

} // namespace gl
} // namespace scm
