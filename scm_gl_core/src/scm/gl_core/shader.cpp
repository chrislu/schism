
#include "shader.h"

#include <cassert>
#include <string>
#include <sstream>

#include <boost/utility.hpp>

#include <scm/core/pointer_types.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/opengl/config.h>
#include <scm/gl_core/opengl/gl3_core.h>
#include <scm/gl_core/opengl/util/assert.h>
#include <scm/gl_core/opengl/util/error_helper.h>

namespace  {

int gl_shader_types[] = {
    0,                  // TYPE_UNKNOWN,           = 0x00,
    GL_VERTEX_SHADER,   // TYPE_VERTEX_SHADER      = 0x01,
    GL_GEOMETRY_SHADER, // TYPE_GEOMETRY_SHADER,
    GL_FRAGMENT_SHADER  // TYPE_FRAGMENT_SHADER
};

} // namespace 

namespace scm {
namespace gl {

shader::stage_type
shader::type() const
{
    return (_type);
}

const std::string&
shader::info_log() const
{
    return (_info_log);
}

shader::shader(render_device&       ren_dev,
               stage_type           in_type,
               const std::string&   in_src)
  : render_device_child(ren_dev),
    _type(in_type),
    _gl_shader_obj(0)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

    _gl_shader_obj = glapi.glCreateShader(gl_shader_types[in_type]);
    if (0 == _gl_shader_obj) {
        state().set(object_state::OS_BAD);
    }
    else {
        compile_source_string(ren_dev, in_src);
    }
    
    gl_assert(glapi, leaving shader::shader());
}

shader::~shader()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_shader_obj);
    glapi.glDeleteShader(_gl_shader_obj);
    
    gl_assert(glapi, leaving shader::~shader());
}

bool
shader::compile_source_string(      render_device&  ren_dev,
                              const std::string&    in_src)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

    const char* source_string = in_src.c_str();
    glapi.glShaderSource(_gl_shader_obj, 1, reinterpret_cast<const GLchar**>(boost::addressof(source_string)), NULL);
    glapi.glCompileShader(_gl_shader_obj);

    int compile_state = 0;
    glapi.glGetShaderiv(_gl_shader_obj, GL_COMPILE_STATUS, &compile_state);

    if (GL_TRUE != compile_state) {
        state().set(object_state::OS_ERROR_SHADER_COMPILE);
    }

    int info_len = 0;
    glapi.glGetShaderiv(_gl_shader_obj, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len > 10) {
        _info_log.clear();
        _info_log.resize(info_len);

        assert(_info_log.capacity() >= info_len);

        glapi.glGetShaderInfoLog(_gl_shader_obj, info_len, NULL, &_info_log[0]);
    }

    return (GL_TRUE == compile_state);
}

} // namespace gl
} // namespace scm
