
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "program_object.h"

#include <cassert>

#include <scm/gl_classic/opengl.h>
#include <scm/gl_classic/shader_objects/shader_object.h>

namespace scm {
namespace gl_classic {

program_object::binding_guard::binding_guard()
  : _save_current_program(0)
{
    assert(glGetError() == GL_NONE);

    glGetIntegerv(GL_CURRENT_PROGRAM, &_save_current_program);

    assert(glGetError() == GL_NONE);
}

program_object::binding_guard::~binding_guard()
{
    assert(glGetError() == GL_NONE);

    glUseProgram(_save_current_program);

    assert(glGetError() == GL_NONE);
}

program_object::program_object()
  : _prog(new unsigned int),
    _ok(false)
{
    *_prog = glCreateProgram();

    assert(*_prog != 0);
}

program_object::program_object(const program_object& prog_obj)
  : _prog(prog_obj._prog),
    _ok(prog_obj._ok)
{
}

program_object::~program_object()
{
    if (_prog.unique()) {
        glDeleteProgram(*_prog);
    }
}

program_object& program_object::operator=(const program_object& rhs)
{
    if (_prog.unique()) {
        glDeleteProgram(*_prog);
    }

    _prog = rhs._prog;
    _ok   = rhs._ok;

    return (*this);
}

bool program_object::attach_shader(const shader_object& sobj)
{
    glAttachShader(*_prog, sobj._obj);

    if (glGetError() != GL_NONE) {
        return (false);
    }

    return (true);
}

bool program_object::link()
{
    int link_state  = 0;
    _ok             = true;

    glLinkProgram(*_prog);
    glGetProgramiv(*_prog, GL_LINK_STATUS, &link_state);

    // get the linker output
    int info_len = 0;
    glGetProgramiv(*_prog, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len) {
        boost::scoped_array<GLchar> linker_info;
        linker_info.reset(new GLchar[info_len]);
        glGetProgramInfoLog(*_prog, info_len, NULL, linker_info.get());
        _linker_out = std::string(linker_info.get());
    }

    if (!link_state) {
        _ok = false;
    }

    return (_ok);
}

bool program_object::validate()
{
    int valid_state = 0;
    _ok             = true;

    glValidateProgram(*_prog);
    glGetProgramiv(*_prog, GL_VALIDATE_STATUS, &valid_state);

    if (!valid_state) {
        GLchar*   valid_info;
        int       info_len;

        glGetProgramiv(*_prog, GL_INFO_LOG_LENGTH, &info_len);
        valid_info = new GLchar[info_len];
        glGetProgramInfoLog(*_prog, info_len, NULL, valid_info);

        _validate_out = std::string(valid_info);
        delete [] valid_info;

        _ok = false;
    }

    return (_ok);
}

void program_object::bind() const
{
    assert(glGetError() == GL_NONE);

    glUseProgram(*_prog);

    assert(glGetError() == GL_NONE);
}

void program_object::unbind() const
{
    assert(glGetError() == GL_NONE);
    assert(_prog);

    glUseProgram(0);

    assert(glGetError() == GL_NONE);
}

unsigned int program_object::program_id() const
{
    return (*_prog);
}

void program_object::uniform_1f(const std::string& param_name, float x) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform1fEXT(*_prog, location, x);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform1f(location, x);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_2f(const std::string& param_name, float x, float y) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform2fEXT(*_prog, location, x, y);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform2f(location, x, y);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_3f(const std::string& param_name, float x, float y, float z) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform3fEXT(*_prog, location, x, y, z);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform3f(location, x, y, z);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_4f(const std::string& param_name, float x, float y, float z, float w) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform4fEXT(*_prog, location, x, y, z, w);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform4f(location, x, y, z, w);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_1fv(const std::string& param_name, unsigned int count, const float* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform1fvEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform1fv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_2fv(const std::string& param_name, unsigned int count, const float* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform2fvEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform2fv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_3fv(const std::string& param_name, unsigned int count, const float* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform3fvEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform3fv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_4fv(const std::string& param_name, unsigned int count, const float* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform4fvEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform4fv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_1i(const std::string& param_name, int x) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform1iEXT(*_prog, location, x);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform1i(location, x);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_2i(const std::string& param_name, int x, int y) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform2iEXT(*_prog, location, x, y);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform2i(location, x, y);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_3i(const std::string& param_name, int x, int y, int z) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform3iEXT(*_prog, location, x, y, z);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform3i(location, x, y, z);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_4i(const std::string& param_name, int x, int y, int z, int w) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform4iEXT(*_prog, location, x, y, z, w);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform4i(location, x, y, z, w);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_1iv(const std::string& param_name, unsigned int count, const int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform1ivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform1iv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_2iv(const std::string& param_name, unsigned int count, const int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform2ivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform2iv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_3iv(const std::string& param_name, unsigned int count, const int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform3ivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform3iv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_4iv(const std::string& param_name, unsigned int count, const int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform4ivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform4iv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_1ui(const std::string& param_name, unsigned int x) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform1uiEXT(*_prog, location, x);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform1ui(location, x);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_2ui(const std::string& param_name, unsigned int x, unsigned int y) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform2uiEXT(*_prog, location, x, y);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform2ui(location, x, y);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_3ui(const std::string& param_name, unsigned int x, unsigned int y, unsigned int z) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform3uiEXT(*_prog, location, x, y, z);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform3ui(location, x, y, z);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_4ui(const std::string& param_name, unsigned int x, unsigned int y, unsigned int z, unsigned int w) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform4uiEXT(*_prog, location, x, y, z, w);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform4ui(location, x, y, z, w);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_1uiv(const std::string& param_name, unsigned int count, const unsigned int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform1uivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform1uiv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_2uiv(const std::string& param_name, unsigned int count, const unsigned int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform2uivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform2uiv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_3uiv(const std::string& param_name, unsigned int count, const unsigned int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform3uivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform3uiv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_4uiv(const std::string& param_name, unsigned int count, const unsigned int* v) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniform4uivEXT(*_prog, location, count, v);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniform4uiv(location, count, v);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_matrix_2fv(const std::string& param_name, unsigned int count, bool transpose, const float* m) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniformMatrix2fvEXT(*_prog, location, count, transpose, m);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniformMatrix2fv(location, count, transpose, m);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_matrix_3fv(const std::string& param_name, unsigned int count, bool transpose, const float* m) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniformMatrix3fvEXT(*_prog, location, count, transpose, m);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniformMatrix3fv(location, count, transpose, m);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

void program_object::uniform_matrix_4fv(const std::string& param_name, unsigned int count, bool transpose, const float* m) const
{
    assert(_prog);

    int location = uniform_location(param_name);

    if (location > -1) {
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
        glProgramUniformMatrix4fvEXT(*_prog, location, count, transpose, m);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
        binding_guard guard;
        bind();
        glUniformMatrix4fv(location, count, transpose, m);
        unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    }
    assert(glGetError() == GL_NONE);
}

int program_object::uniform_location(const std::string& param_name) const
{
    assert(glGetError() == GL_NONE);
    return (glGetUniformLocation(*_prog, param_name.c_str()));
}

} // namespace gl_classic
} // namespace scm
