
#ifndef VEC_SERIALIZATION_H_INCLUDED
#define VEC_SERIALIZATION_H_INCLUDED

// cool stuff to try, but totally over the top!
#if 0
//#include <istream>
//#include <ostream>

//#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/nvp.hpp>


namespace boost {
namespace serialization {

// math::vec objects are generally serializable
template<typename scm_scalar,
         unsigned dim>
struct implementation_level<math::vec<scm_scalar, dim> >
{
    typedef mpl::integral_c_tag             tag;
    typedef mpl::int_<object_serializable>  type;
    BOOST_STATIC_CONSTANT(
        int,
        value = implementation_level::type::value
    );
};

// math::vec objects are generally created on the stack and are never tracked
template<typename scm_scalar,
         unsigned dim>
struct tracking_level<math::vec<scm_scalar, dim> >
{
    typedef mpl::integral_c_tag     tag;
    typedef mpl::int_<track_never>  type;
    BOOST_STATIC_CONSTANT(
        int, 
        value = tracking_level::type::value
    );
};
template<class      archive,
         typename   scm_scalar,
         unsigned   dim>
         void serialize(archive& ar, math::vec<scm_scalar, dim>& vec, const unsigned int version)
{
    ar & make_nvp("vector", vec.vec_array);
}

} // namespace serialize
} // namespace boost
#endif

namespace math {

template<typename scm_scalar,
         unsigned dim>
std::ostream& operator<<(std::ostream& out_stream, const vec<scm_scalar, dim>& out_vec)
{
    std::ostream::sentry        out_sentry(out_stream);

    if (!out_sentry) {
        out_stream.setstate(std::ios_base::failbit);
        return (out_stream);
    }

    out_stream << "(";
    for (std::size_t i = 0; i < dim; ++i) {
        out_stream << (i != 0 ? "  " : "") << out_vec.vec_array[i];
    }
    out_stream << ")";

    return (out_stream);
}

template<typename scm_scalar, unsigned dim>
std::istream& operator>>(std::istream& in_stream, math::vec<scm_scalar, dim>& in_vec)
{
    std::istream::sentry        in_sentry(in_stream);

    if (!in_sentry) {
        in_stream.setstate(std::ios_base::failbit);
        return (in_stream);
    }

    math::vec<scm_scalar, dim>  tmp_vec;
    std::istream::char_type     cur_char(0);
    bool                        bracket_version(false);

    in_stream >> cur_char;

    bracket_version = (cur_char == std::istream::char_type('('));

    if (!bracket_version) {
        in_stream.putback(cur_char);
    }
    for (std::size_t i = 0; i < dim; ++i) {
        in_stream >> tmp_vec.vec_array[i];
    }
    if (bracket_version) {
        in_stream >> cur_char;
        if (cur_char != std::istream::char_type(')')) {
            in_stream.clear(std::ios_base::badbit);
        }
    }

    if (in_stream) {
        in_vec = tmp_vec;
    }

    return (in_stream);
}

} // namespace math

#endif // VEC_SERIALIZATION_H_INCLUDED
