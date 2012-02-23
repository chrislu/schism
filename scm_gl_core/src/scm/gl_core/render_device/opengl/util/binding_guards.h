
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_BINDING_GUARDS_H_INCLUDED
#define SCM_GL_CORE_BINDING_GUARDS_H_INCLUDED

#include <scm/core/math.h>

namespace scm {
namespace gl {

namespace opengl {
class gl_core;
}

namespace util {

class texture_binding_guard
{
public:
    explicit texture_binding_guard(const opengl::gl_core& in_glapi,
                          unsigned                in_target,
                          unsigned                in_binding);
    virtual ~texture_binding_guard();
private:
    int                     _save_active_texture_unit;
    int                     _save_texture_object;
    unsigned                _binding;
    unsigned                _target;
    const opengl::gl_core& _gl_api;
};

class program_binding_guard
{
public:
    explicit program_binding_guard(const opengl::gl_core& in_glapi);
    virtual ~program_binding_guard();
private:
    int                     _active_program;
    const opengl::gl_core& _gl_api;
};

class buffer_binding_guard
{
public:
    explicit buffer_binding_guard(const opengl::gl_core& in_glapi,
                         unsigned                in_target,
                         unsigned                in_binding);
    virtual ~buffer_binding_guard();
private:
    int             _save;
    unsigned        _binding;
    unsigned        _target;
    const opengl::gl_core& _gl_api;
};

class vertex_array_binding_guard
{
public:
    explicit vertex_array_binding_guard(const opengl::gl_core& in_glapi);
    virtual ~vertex_array_binding_guard();
private:
    int             _save;
    const opengl::gl_core& _gl_api;
};

class framebuffer_binding_guard
{
public:
    explicit framebuffer_binding_guard(const opengl::gl_core& in_glapi,
                                       unsigned                in_target,
                                       unsigned                in_binding);
    virtual ~framebuffer_binding_guard();
private:
    int             _save;
    unsigned        _target;
    unsigned        _binding;
    const opengl::gl_core& _gl_api;
};

class transform_feedback_binding_guard
{
public:
    explicit transform_feedback_binding_guard(const opengl::gl_core& in_glapi,
                                              unsigned                in_target,
                                              unsigned                in_binding);
    virtual ~transform_feedback_binding_guard();
private:
    int             _save;
    unsigned        _target;
    unsigned        _binding;
    const opengl::gl_core& _gl_api;
};

} // namespace util
} // namespace gl
} // namespace scm


#endif // SCM_GL_CORE_BINDING_GUARDS_H_INCLUDED
