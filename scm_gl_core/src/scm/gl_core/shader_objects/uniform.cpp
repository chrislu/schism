
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "uniform.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/shader_objects/program.h>

namespace scm {
namespace gl {

template<typename T, data_type D>
uniform<T, D>::uniform(const std::string& n, const int l, const unsigned e, const data_type t)
  : uniform_base(n, l, e, t)
  , _value(e)
{
    assert(e > 0);
}

template<typename T, data_type D>
uniform<T, D>::~uniform()
{
}

template<typename T, data_type D>
typename uniform<T, D>::value_param_type
uniform<T, D>::value() const
{
    return value(0);
}

template<typename T, data_type D>
void
uniform<T, D>::set_value(value_param_type v)
{
    set_value(0, v);
}

template<typename T, data_type D>
typename uniform<T, D>::value_param_type
uniform<T, D>::value(int i) const
{
    assert(i < static_cast<int>(_elements));
    return _value[i];
}

template<typename T, data_type D>
void
uniform<T, D>::set_value(int i, value_param_type v)
{
    assert(i < static_cast<int>(_elements));
    if (!_status._initialized || v != _value[i]) {
        _value[i] = v;
        _status._update_required = true;
        _status._initialized     = true;
    }
}

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

// instantiate the uniform templates //////////////////////////////////////////////////////////////
#define SCM_UNIFORM_TYPE_INSTANTIATE(type_raw, type_id, uniform_type_name)               \
    template class __scm_export(gl_core)                 scm::gl::uniform<type_raw, type_id>;

//namespace scm {
//namespace gl {

SCM_UNIFORM_TYPE_INSTANTIATE(float,             scm::gl::TYPE_FLOAT,  uniform_1f)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec2f,  scm::gl::TYPE_VEC2F,  uniform_vec2f)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec3f,  scm::gl::TYPE_VEC3F,  uniform_vec3f)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec4f,  scm::gl::TYPE_VEC4F,  uniform_vec4f)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat2f,  scm::gl::TYPE_MAT2F,  uniform_mat2f)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat3f,  scm::gl::TYPE_MAT3F,  uniform_mat3f)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat4f,  scm::gl::TYPE_MAT4F,  uniform_mat4f)

SCM_UNIFORM_TYPE_INSTANTIATE(int,               scm::gl::TYPE_INT,    uniform_1i)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec2i,  scm::gl::TYPE_VEC2I,  uniform_vec2i)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec3i,  scm::gl::TYPE_VEC3I,  uniform_vec3i)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec4i,  scm::gl::TYPE_VEC4I,  uniform_vec4i)

SCM_UNIFORM_TYPE_INSTANTIATE(unsigned,          scm::gl::TYPE_UINT,   uniform_1ui)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec2ui, scm::gl::TYPE_VEC2UI, uniform_vec2ui)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec3ui, scm::gl::TYPE_VEC3UI, uniform_vec3ui)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec4ui, scm::gl::TYPE_VEC4UI, uniform_vec4ui)

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
SCM_UNIFORM_TYPE_INSTANTIATE(double,            scm::gl::TYPE_DOUBLE, uniform_1d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec2d,  scm::gl::TYPE_VEC2D,  uniform_vec2d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec3d,  scm::gl::TYPE_VEC3D,  uniform_vec3d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec4d,  scm::gl::TYPE_VEC4D,  uniform_vec4d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat2d,  scm::gl::TYPE_MAT2D,  uniform_mat2d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat3d,  scm::gl::TYPE_MAT3D,  uniform_mat3d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat4d,  scm::gl::TYPE_MAT4D,  uniform_mat4d)
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

//} // namespace gl
//} // namespace scm

#undef SCM_UNIFORM_TYPE_INSTANTIATE

namespace scm {
namespace gl {

uniform_base::uniform_base(const std::string& n, const int l, const unsigned e, const data_type t)
  : _name(n)
  , _location(l)
  , _elements(e)
  , _type(t)
{
    _status._initialized     = false;
    _status._update_required = false;
}

uniform_base::~uniform_base()
{
}

const std::string&
uniform_base::name() const
{
    return _name;
}

const int
uniform_base::location() const
{
    return _location;
}

const unsigned
uniform_base::elements() const
{
    return _elements;
}

const data_type
uniform_base::type() const
{
    return _type;
}

bool
uniform_base::update_required() const
{
    return _status._update_required;
}

// class uniform_image_sampler_base ///////////////////////////////////////////////////////////////
uniform_image_sampler_base::uniform_image_sampler_base(const std::string& n, const int l, const unsigned e, const data_type t)
  : uniform_base(n, l, e, t)
  , _bound_unit(-1)
  , _resident_handle(0ull)
{
}

uniform_image_sampler_base::~uniform_image_sampler_base()
{
}

scm::int32
uniform_image_sampler_base::bound_unit() const
{
    return _bound_unit;
}

void
uniform_image_sampler_base::bound_unit(scm::int32 v)
{
    if (!_status._initialized || v != _bound_unit) {
        _bound_unit              = v;
        _resident_handle         = 0ull;
        _status._update_required = true;
        _status._initialized     = true;
    }
}

scm::uint64
uniform_image_sampler_base::resident_handle()
{
    return _resident_handle;
}

void
uniform_image_sampler_base::resident_handle(scm::uint64 v)
{
    if (!_status._initialized || v != _resident_handle) {
        _bound_unit              = -1;
        _resident_handle         = v;
        _status._update_required = true;
        _status._initialized     = true;
    }
}

void
uniform_image_sampler_base::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
#if SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    if (_bound_unit >= 0) {
        glapi.glProgramUniform1i(p.program_id(), _location, _bound_unit);
    }
    else if (   glapi.extension_ARB_bindless_texture
             && _resident_handle != 0ull)
    {
        glapi.glProgramUniformHandleui64ARB(p.program_id(), _location, _resident_handle);
    }
#else
    if (_bound_unit >= 0) {
        glapi.glUniform1i(_location, _bound_unit);
    }
    else if (   glapi.extension_ARB_bindless_texture
             && _resident_handle != 0ull)
    {
        glapi.glUniformHandleui64ARB(_location, _resident_handle);
    }
#endif
    gl_assert(glapi, leaving uniform_1f::apply_value());

}

// float types ////////////////////////////////////////////////////////////////////////////////////
template<>
void
uniform_1f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform1fv(_location, _elements, &(_value.front()));
    gl_assert(glapi, leaving uniform_1f::apply_value());
}

template<>
void
uniform_vec2f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform2fv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec2f::apply_value());
}

template<>
void
uniform_vec3f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform3fv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec3f::apply_value());
}

template<>
void
uniform_vec4f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform4fv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec4f::apply_value());
}

template<>
void
uniform_mat2f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniformMatrix2fv(_location, _elements, false, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_mat2f::apply_value());
}

template<>
void
uniform_mat3f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniformMatrix3fv(_location, _elements, false, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_mat3f::apply_value());
}

template<>
void
uniform_mat4f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniformMatrix4fv(_location, _elements, false, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_mat4f::apply_value());
}

// double types ///////////////////////////////////////////////////////////////////////////////////
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
template<>
void
uniform_1d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform1dv(_location, _elements, &(_value.front()));//&_value);
    gl_assert(glapi, leaving uniform_1d::apply_value());
}

template<>
void
uniform_vec2d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform2dv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec2d::apply_value());
}

template<>
void
uniform_vec3d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform3dv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec3d::apply_value());
}

template<>
void
uniform_vec4d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform4dv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec4d::apply_value());
}

template<>
void
uniform_mat2d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniformMatrix2dv(_location, _elements, false, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_mat2d::apply_value());
}

template<>
void
uniform_mat3d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniformMatrix3dv(_location, _elements, false, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_mat3d::apply_value());
}

template<>
void
uniform_mat4d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniformMatrix4dv(_location, _elements, false, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_mat4d::apply_value());
}

#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

// int types //////////////////////////////////////////////////////////////////////////////////////
template<>
void
uniform_1i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform1iv(_location, _elements, &(_value.front()));//&_value);
    gl_assert(glapi, leaving uniform_1i::apply_value());
}

template<>
void
uniform_vec2i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform2iv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec2i::apply_value());
}

template<>
void
uniform_vec3i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform3iv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec3i::apply_value());
}

template<>
void
uniform_vec4i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform4iv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec4i::apply_value());
}

// uint types /////////////////////////////////////////////////////////////////////////////////////
template<>
void
uniform_1ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform1uiv(_location, _elements, &(_value.front()));//&_value);
    gl_assert(glapi, leaving uniform_1ui::apply_value());
}

template<>
void
uniform_vec2ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform2uiv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec2ui::apply_value());
}

template<>
void
uniform_vec3ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform3uiv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec3ui::apply_value());
}

template<>
void
uniform_vec4ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl_core& glapi = context.opengl_api();
    glapi.glUniform4uiv(_location, _elements, (_value.front().data_array));//_value.data_array);
    gl_assert(glapi, leaving uniform_vec4ui::apply_value());
}

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

