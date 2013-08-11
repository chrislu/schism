
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <scm/gl_core/log.h>

#include <boost/mpl/find.hpp>
#include <boost/type_traits/is_same.hpp>

namespace scm {
namespace gl {

template<typename T>
inline
void
program::uniform(const std::string& name, const T& v) const {
    uniform(name, 0, v);
}

template<typename T>
inline
void
program::uniform(const std::string& name, int i, const T& v) const {
    const uniform_ptr u = uniform_raw(name);
    if (u) {
        typedef typename scm::gl::uniform_type<T>::type cur_uniform_type;
        const shared_ptr<cur_uniform_type> ut = dynamic_pointer_cast<cur_uniform_type>(u);
        if (ut) {
            ut->set_value(i, v);
        }
        else {
            SCM_GL_DGB("program::uniform(): found non matching uniform type '" << type_string(uniform_data_type<T>::type)
                                                                               << "' ('uniform: " << name << ", " << type_string(u->type()) << ").");
            //SCM_GL_LOG_ONCE(log::warning, "program::uniform(): found non matching uniform type '"
            //                              << type_string(uniform_base_type<T>::type)
            //                              << "' ('uniform: " << name << ", " << type_string(u->type()) << ").");
        }
    }
    else {
        SCM_GL_DGB("program::uniform(): unable to find uniform ('" << name << "').");
        //SCM_GL_LOG_ONCE(log::warning, "program::uniform(): unable to find uniform ('" << name << "').");
    }
}

inline uniform_sampler_ptr
program::uniform_sampler(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_sampler>(uniform_raw(name)));
}

inline uniform_image_ptr
program::uniform_image(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_image>(uniform_raw(name)));
}

inline uniform_1f_ptr
program::uniform_1f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_1f>(uniform_raw(name)));
}

inline uniform_vec2f_ptr
program::uniform_vec2f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec2f>(uniform_raw(name)));
}

inline uniform_vec3f_ptr
program::uniform_vec3f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec3f>(uniform_raw(name)));
}

inline uniform_vec4f_ptr
program::uniform_vec4f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec4f>(uniform_raw(name)));
}

inline uniform_mat2f_ptr
program::uniform_mat2f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_mat2f>(uniform_raw(name)));
}

inline uniform_mat3f_ptr
program::uniform_mat3f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_mat3f>(uniform_raw(name)));
}

inline uniform_mat4f_ptr
program::uniform_mat4f(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_mat4f>(uniform_raw(name)));
}

inline uniform_1i_ptr
program::uniform_1i(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_1i>(uniform_raw(name)));
}

inline uniform_vec2i_ptr
program::uniform_vec2i(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec2i>(uniform_raw(name)));
}

inline uniform_vec3i_ptr
program::uniform_vec3i(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec3i>(uniform_raw(name)));
}

inline uniform_vec4i_ptr
program::uniform_vec4i(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec4i>(uniform_raw(name)));
}

inline uniform_1ui_ptr
program::uniform_1ui(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_1ui>(uniform_raw(name)));
}

inline uniform_vec2ui_ptr
program::uniform_vec2ui(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec2ui>(uniform_raw(name)));
}

inline uniform_vec3ui_ptr
program::uniform_vec3ui(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec3ui>(uniform_raw(name)));
}

inline uniform_vec4ui_ptr
program::uniform_vec4ui(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec4ui>(uniform_raw(name)));
}

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
inline uniform_1d_ptr
program::uniform_1d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_1d>(uniform_raw(name)));
}

inline uniform_vec2d_ptr
program::uniform_vec2d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec2d>(uniform_raw(name)));
}

inline uniform_vec3d_ptr
program::uniform_vec3d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec3d>(uniform_raw(name)));
}

inline uniform_vec4d_ptr
program::uniform_vec4d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_vec4d>(uniform_raw(name)));
}

inline uniform_mat2d_ptr
program::uniform_mat2d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_mat2d>(uniform_raw(name)));
}

inline uniform_mat3d_ptr
program::uniform_mat3d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_mat3d>(uniform_raw(name)));
}

inline uniform_mat4d_ptr
program::uniform_mat4d(const std::string& name) const {
    return (dynamic_pointer_cast<scm::gl::uniform_mat4d>(uniform_raw(name)));
}
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

namespace detail {

//template<typename lhs_type>
//class uniform_equals : public boost::static_visitor<bool>
//{
//    const lhs_type& _lhs;
//public:
//    uniform_equals(const lhs_type& lhs) : _lhs(lhs) {}
//    template <typename T>
//    bool operator()(const T&) const             { return (false); }         // cannot compare different types
//    bool operator()(const lhs_type& rhs) const  { return (_lhs == rhs); }
//};
//
//} // namespace detail
//
//template<typename utype>
//void program::uniform(const std::string& name, const utype& value)
//{
//    typedef typename boost::mpl::find<scm::gl::uniform_types, utype>::type type_iter;
//    BOOST_MPL_ASSERT((boost::is_same<typename boost::mpl::deref<type_iter>::type, utype>));
//
//    name_uniform_map::iterator  u = _uniforms.find(name);
//
//    if (u != _uniforms.end()) {
//        if (uniform_type_id<utype>::id == u->second._type) {
//            if (!boost::apply_visitor(detail::uniform_equals<utype>(value), u->second._data)) {
//                u->second._data = value;
//                u->second._update_required = true;
//            }
//        }
//        else {
//            SCM_GL_DGB("program::uniform(): found non matching uniform type '" << type_string(uniform_type_id<utype>::id) << "' ('uniform: " << name << ", " << type_string(u->second._type) << ").");
//        }
//    }
//    else {
//       // SCM_GL_DGB("program::uniform(): unable to find uniform ('" << name << "').");
//    }

} // namespace detail

} // namespace gl
} // namespace scm
