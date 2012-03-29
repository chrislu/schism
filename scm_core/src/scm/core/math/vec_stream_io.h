
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_VEC_STREAM_IO_H_INCLUDED
#define MATH_VEC_STREAM_IO_H_INCLUDED

#include <iomanip>
#include <ostream>
#include <istream>
#include <boost/io/ios_state.hpp>

namespace scm {
namespace math {

template<typename scal_type,
         const unsigned dim>
std::ostream& operator<<(std::ostream& out_stream, const vec<scal_type, dim>& out_vec)
{
    std::ostream::sentry const  out_sentry(out_stream);

    if (out_sentry) {
        boost::io::ios_all_saver saved_state(out_stream);

        out_stream << std::fixed << std::setprecision(3);

        out_stream << "(";
        for (unsigned i = 0; i < dim; ++i) {
            out_stream << (i != 0 ? "  " : "") << out_vec.data_array[i];
        }
        out_stream << ")";
    }
    else {
        out_stream.setstate(std::ios_base::failbit);
    }

    return (out_stream);
}

template<typename scal_type,
         const unsigned dim>
std::istream& operator>>(std::istream& in_stream, vec<scal_type, dim>& in_vec)
{
    std::istream::sentry const  in_sentry(in_stream);

    if (in_sentry) {
        vec<scal_type, dim>         tmp_vec;
        std::istream::char_type     cur_char(0);
        bool                        bracket_version(false);

        in_stream >> cur_char;

        bracket_version = (cur_char == std::istream::char_type('('));

        if (!bracket_version) {
            in_stream.putback(cur_char);
        }
        for (unsigned i = 0; i < dim; ++i) {
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
    }
    else {
        in_stream.setstate(std::ios_base::failbit);
    }

    return (in_stream);
}

} // namespace math
} // namespace scm

#endif // MATH_VEC_STREAM_IO_H_INCLUDED
