
#include "uniform.h"

#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

template<typename T, data_type D>
uniform<T, D>::uniform(const std::string& n, const int l, const unsigned e, const data_type t)
  : uniform_base(n, l, e, t)
{
}

template<typename T, data_type D>
uniform<T, D>::~uniform()
{
}

template<typename T, data_type D>
typename uniform<T, D>::value_param_type
uniform<T, D>::value() const
{
    return (_value);
}

template<typename T, data_type D>
void
uniform<T, D>::value(value_param_type v)
{
    if (v != _value) {
        _value = v;
        _update_required = true;
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

#if SCM_GL_CORE_OPENGL_40
SCM_UNIFORM_TYPE_INSTANTIATE(double,            scm::gl::TYPE_DOUBLE, uniform_1d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec2d,  scm::gl::TYPE_VEC2D,  uniform_vec2d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec3d,  scm::gl::TYPE_VEC3D,  uniform_vec3d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::vec4d,  scm::gl::TYPE_VEC4D,  uniform_vec4d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat2d,  scm::gl::TYPE_MAT2D,  uniform_mat2d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat3d,  scm::gl::TYPE_MAT3D,  uniform_mat3d)
SCM_UNIFORM_TYPE_INSTANTIATE(scm::math::mat4d,  scm::gl::TYPE_MAT4D,  uniform_mat4d)
#endif // SCM_GL_CORE_OPENGL_40

//} // namespace gl
//} // namespace scm

#undef SCM_UNIFORM_TYPE_INSTANTIATE

namespace scm {
namespace gl {

uniform_base::uniform_base(const std::string& n, const int l, const unsigned e, const data_type t)
  : _name(n)
  , _location(l)
  , _elements(e)
  , _update_required(false)
  , _type(t)
{
}

uniform_base::~uniform_base()
{
}

const std::string&
uniform_base::name() const
{
    return (_name);
}

const int
uniform_base::location() const
{
    return (_location);
}

const unsigned
uniform_base::elements() const
{
    return (_elements);
}

const data_type
uniform_base::type() const
{
    return (_type);
}

bool
uniform_base::update_required() const
{
    return (_update_required);
}

// float types ////////////////////////////////////////////////////////////////////////////////////
template<>
void
uniform_1f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform1fv(_location, /*_elements*/ 1, &_value);
    gl_assert(glapi, leaving uniform_1f::apply_value());
}

template<>
void
uniform_vec2f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform2fv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec2f::apply_value());
}

template<>
void
uniform_vec3f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform3fv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec3f::apply_value());
}

template<>
void
uniform_vec4f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform4fv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec4f::apply_value());
}

template<>
void
uniform_mat2f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniformMatrix2fv(_location, /*_elements*/ 1, false, _value.data_array);
    gl_assert(glapi, leaving uniform_mat2f::apply_value());
}

template<>
void
uniform_mat3f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniformMatrix3fv(_location, /*_elements*/ 1, false, _value.data_array);
    gl_assert(glapi, leaving uniform_mat3f::apply_value());
}

template<>
void
uniform_mat4f::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniformMatrix4fv(_location, /*_elements*/ 1, false, _value.data_array);
    gl_assert(glapi, leaving uniform_mat4f::apply_value());
}

// double types ///////////////////////////////////////////////////////////////////////////////////
#if SCM_GL_CORE_OPENGL_40
template<>
void
uniform_1d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform1dv(_location, /*_elements*/ 1, &_value);
    gl_assert(glapi, leaving uniform_1d::apply_value());
}

template<>
void
uniform_vec2d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform2dv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec2d::apply_value());
}

template<>
void
uniform_vec3d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform3dv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec3d::apply_value());
}

template<>
void
uniform_vec4d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform4dv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec4d::apply_value());
}

template<>
void
uniform_mat2d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniformMatrix2dv(_location, /*_elements*/ 1, false, _value.data_array);
    gl_assert(glapi, leaving uniform_mat2d::apply_value());
}

template<>
void
uniform_mat3d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniformMatrix3dv(_location, /*_elements*/ 1, false, _value.data_array);
    gl_assert(glapi, leaving uniform_mat3d::apply_value());
}

template<>
void
uniform_mat4d::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniformMatrix4dv(_location, /*_elements*/ 1, false, _value.data_array);
    gl_assert(glapi, leaving uniform_mat4d::apply_value());
}


#endif // SCM_GL_CORE_OPENGL_40

// int types //////////////////////////////////////////////////////////////////////////////////////
template<>
void
uniform_1i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform1iv(_location, /*_elements*/ 1, &_value);
    gl_assert(glapi, leaving uniform_1i::apply_value());
}

template<>
void
uniform_vec2i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform2iv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec2i::apply_value());
}

template<>
void
uniform_vec3i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform3iv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec3i::apply_value());
}

template<>
void
uniform_vec4i::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform4iv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec4i::apply_value());
}

// uint types /////////////////////////////////////////////////////////////////////////////////////
template<>
void
uniform_1ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform1uiv(_location, /*_elements*/ 1, &_value);
    gl_assert(glapi, leaving uniform_1ui::apply_value());
}

template<>
void
uniform_vec2ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform2uiv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec2ui::apply_value());
}

template<>
void
uniform_vec3ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform3uiv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec3ui::apply_value());
}

template<>
void
uniform_vec4ui::apply_value(const render_context& context, const program& p)
{
    const opengl::gl3_core& glapi = context.opengl_api();
    glapi.glUniform4uiv(_location, /*_elements*/ 1, _value.data_array);
    gl_assert(glapi, leaving uniform_vec4ui::apply_value());
}


} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

