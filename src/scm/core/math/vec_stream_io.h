
#ifndef VEC_STREAM_IO_H_INCLUDED
#define VEC_STREAM_IO_H_INCLUDED

namespace scm {
namespace math {

template<typename scm_scalar, unsigned dim>
std::ostream& operator<<(std::ostream& out_stream, const vec<scm_scalar, dim>& out_vec)
{
    std::ostream::sentry        out_sentry(out_stream);

    if (!out_sentry) {
        out_stream.setstate(std::ios_base::failbit);
        return (out_stream);
    }

    out_stream << "(";
    for (std::size_t i = 0; i < dim; ++i) {
        out_stream << (i != 0 ? "  " : "") << out_vec.data_array[i];
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
        in_stream >> tmp_vec.data_array[i];
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
} // namespace scm

#endif // VEC_STREAM_IO_H_INCLUDED
