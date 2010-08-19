
#include <scm/gl_core/log.h>

#include <boost/mpl/find.hpp>
#include <boost/type_traits/is_same.hpp>

namespace scm {
namespace gl {

namespace detail {

template<typename lhs_type>
class uniform_equals : public boost::static_visitor<bool>
{
    const lhs_type& _lhs;
public:
    uniform_equals(const lhs_type& lhs) : _lhs(lhs) {}
    template <typename T>
    bool operator()(const T&) const             { return (false); }         // cannot compare different types
    bool operator()(const lhs_type& rhs) const  { return (_lhs == rhs); }
};

} // namespace detail

template<typename utype>
void program::uniform(const std::string& name, const utype& value)
{
    typedef typename boost::mpl::find<scm::gl::uniform_types, utype>::type type_iter;
    BOOST_MPL_ASSERT((boost::is_same<typename boost::mpl::deref<type_iter>::type, utype>));

    name_uniform_map::iterator  u = _uniforms.find(name);

    if (u != _uniforms.end()) {
        if (uniform_type_id<utype>::id == u->second._type) {
            if (!boost::apply_visitor(detail::uniform_equals<utype>(value), u->second._data)) {
                u->second._data = value;
                u->second._update_required = true;
            }
        }
        else {
            SCM_GL_DGB("program::uniform(): found non matching uniform type '" << type_string(uniform_type_id<utype>::id) << "' ('uniform: " << name << ", " << type_string(u->second._type) << ").");
        }
    }
    else {
       // SCM_GL_DGB("program::uniform(): unable to find uniform ('" << name << "').");
    }
}

} // namespace gl
} // namespace scm
