
#ifndef SCM_GL_CORE_PROGRAM_UNIFORM_TYPES_H_INCLUDED
#define SCM_GL_CORE_PROGRAM_UNIFORM_TYPES_H_INCLUDED

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_VECTOR_SIZE 30
#include <boost/mpl/vector.hpp>
#include <boost/variant.hpp>

#include <scm/core/math.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/data_types.h>

namespace scm {
namespace gl {

typedef boost::mpl::vector<float,    scm::math::vec2f,  scm::math::vec3f,  scm::math::vec4f, scm::math::mat2f, scm::math::mat3f, scm::math::mat4f,
#if SCM_GL_CORE_OPENGL_40
                           double,   scm::math::vec2d,  scm::math::vec3d,  scm::math::vec4d, scm::math::mat2d, scm::math::mat3d, scm::math::mat4d,
#endif
                           int,      scm::math::vec2i,  scm::math::vec3i,  scm::math::vec4i,
                           unsigned, scm::math::vec2ui, scm::math::vec3ui, scm::math::vec4ui> uniform_types;

template<typename utype> struct uniform_type_id {};
template<> struct uniform_type_id<float>            { static const data_type id = TYPE_FLOAT; };
template<> struct uniform_type_id<scm::math::vec2f> { static const data_type id = TYPE_VEC2F; };
template<> struct uniform_type_id<scm::math::vec3f> { static const data_type id = TYPE_VEC3F; };
template<> struct uniform_type_id<scm::math::vec4f> { static const data_type id = TYPE_VEC4F; };
template<> struct uniform_type_id<scm::math::mat2f> { static const data_type id = TYPE_MAT2F; };
template<> struct uniform_type_id<scm::math::mat3f> { static const data_type id = TYPE_MAT3F; };
template<> struct uniform_type_id<scm::math::mat4f> { static const data_type id = TYPE_MAT4F; };

#if SCM_GL_CORE_OPENGL_40
template<> struct uniform_type_id<double>           { static const data_type id = TYPE_DOUBLE; };
template<> struct uniform_type_id<scm::math::vec2d> { static const data_type id = TYPE_VEC2D; };
template<> struct uniform_type_id<scm::math::vec3d> { static const data_type id = TYPE_VEC3D; };
template<> struct uniform_type_id<scm::math::vec4d> { static const data_type id = TYPE_VEC4D; };
template<> struct uniform_type_id<scm::math::mat2d> { static const data_type id = TYPE_MAT2D; };
template<> struct uniform_type_id<scm::math::mat3d> { static const data_type id = TYPE_MAT3D; };
template<> struct uniform_type_id<scm::math::mat4d> { static const data_type id = TYPE_MAT4D; };
#endif

template<> struct uniform_type_id<int>              { static const data_type id = TYPE_INT; };
template<> struct uniform_type_id<scm::math::vec2i> { static const data_type id = TYPE_VEC2I; };
template<> struct uniform_type_id<scm::math::vec3i> { static const data_type id = TYPE_VEC3I; };
template<> struct uniform_type_id<scm::math::vec4i> { static const data_type id = TYPE_VEC4I; };

template<> struct uniform_type_id<unsigned>          { static const data_type id = TYPE_UINT; };
template<> struct uniform_type_id<scm::math::vec2ui> { static const data_type id = TYPE_VEC2UI; };
template<> struct uniform_type_id<scm::math::vec3ui> { static const data_type id = TYPE_VEC3UI; };
template<> struct uniform_type_id<scm::math::vec4ui> { static const data_type id = TYPE_VEC4UI; };

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_PROGRAM_UNIFORM_TYPES_H_INCLUDED
