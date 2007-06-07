
#include "program_object.h"

#include <cassert>

#include <scm/ogl/gl.h>
#include <scm/ogl/shader_objects/shader_object.h>

using namespace scm::gl;

program_object::program_object()
    : _prog(0)
{
    _prog = glCreateProgram();

    assert(_prog != 0);
}

program_object::~program_object()
{
    glDeleteProgram(_prog);
}

bool program_object::attach_shader(const shader_object& sobj)
{
    glAttachShader(_prog, sobj._obj);

    if (glGetError() != GL_NONE) {
        return (false);
    }

    return (true);
}

bool program_object::link()
{
    int link_state = 0;

    glLinkProgram(_prog);
    glGetProgramiv(_prog, GL_LINK_STATUS, &link_state);

    if (!link_state) {
        GLchar*   linker_info;
        int       info_len;

        glGetProgramiv(_prog, GL_INFO_LOG_LENGTH, &info_len);
        linker_info = new GLchar[info_len];
        glGetProgramInfoLog(_prog, info_len, NULL, linker_info);

        _linker_out = std::string(linker_info);
        delete [] linker_info;

        return (false);
    }

    return (true);
}

bool program_object::validate()
{
    int valid_state = 0;

    glValidateProgram(_prog);
    glGetProgramiv(_prog, GL_VALIDATE_STATUS, &valid_state);

    if (!valid_state) {
        GLchar*   valid_info;
        int       info_len;

        glGetProgramiv(_prog, GL_INFO_LOG_LENGTH, &info_len);
        valid_info = new GLchar[info_len];
        glGetProgramInfoLog(_prog, info_len, NULL, valid_info);

        _validate_out = std::string(valid_info);
        delete [] valid_info;

        return (false);
    }

    return (true);
}

void program_object::bind() const
{
    glUseProgram(_prog);
}

void program_object::unbind() const
{
    glUseProgram(0);
}

void program_object::set_uniform_1f(const std::string& param_name, float x) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform1f(location, x);
    }
}

void program_object::set_uniform_2f(const std::string& param_name, float x, float y) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform2f(location, x, y);
    }
}

void program_object::set_uniform_3f(const std::string& param_name, float x, float y, float z) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform3f(location, x, y, z);
    }
}

void program_object::set_uniform_4f(const std::string& param_name, float x, float y, float z, float w) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform4f(location, x, y, z, w);
    }
}

void program_object::set_uniform_1fv(const std::string& param_name, unsigned int count, const float* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform1fv(location, count, v);
    }
}

void program_object::set_uniform_2fv(const std::string& param_name, unsigned int count, const float* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform2fv(location, count, v);
    }
}

void program_object::set_uniform_3fv(const std::string& param_name, unsigned int count, const float* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform3fv(location, count, v);
    }
}

void program_object::set_uniform_4fv(const std::string& param_name, unsigned int count, const float* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform4fv(location, count, v);
    }
}

void program_object::set_uniform_1i(const std::string& param_name, int x) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform1i(location, x);
    }
}

void program_object::set_uniform_2i(const std::string& param_name, int x, int y) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform2i(location, x, y);
    }
}

void program_object::set_uniform_3i(const std::string& param_name, int x, int y, int z) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform3i(location, x, y, z);
    }
}

void program_object::set_uniform_4i(const std::string& param_name, int x, int y, int z, int w) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform4i(location, x, y, z, w);
    }
}

void program_object::set_uniform_1iv(const std::string& param_name, unsigned int count, const int* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform1iv(location, count, v);
    }
}

void program_object::set_uniform_2iv(const std::string& param_name, unsigned int count, const int* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform2iv(location, count, v);
    }
}

void program_object::set_uniform_3iv(const std::string& param_name, unsigned int count, const int* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform3iv(location, count, v);
    }
}

void program_object::set_uniform_4iv(const std::string& param_name, unsigned int count, const int* v) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniform4iv(location, count, v);
    }
}

void program_object::set_uniform_matrix_2fv(const std::string& param_name, unsigned int count, bool transpose, const float* m) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniformMatrix2fv(location, count, transpose, m);
    }
}

void program_object::set_uniform_matrix_3fv(const std::string& param_name, unsigned int count, bool transpose, const float* m) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniformMatrix3fv(location, count, transpose, m);
    }
}

void program_object::set_uniform_matrix_4fv(const std::string& param_name, unsigned int count, bool transpose, const float* m) const
{
    int location = get_uniform_location(param_name);

    if (location > -1) {
        glUniformMatrix4fv(location, count, transpose, m);
    }
}

int program_object::get_uniform_location(const std::string& param_name) const
{
    return (glGetUniformLocation(_prog, param_name.c_str()));
}
