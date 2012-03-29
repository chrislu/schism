
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_MAT_STREAM_IO_H_INCLUDED
#define MATH_MAT_STREAM_IO_H_INCLUDED

#include <iomanip>
#include <ostream>
#include <istream>
#include <boost/io/ios_state.hpp>

namespace scm {
namespace math {

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
std::ostream& operator<<(std::ostream& out_stream, const mat<scal_type, row_dim, col_dim>& out_mat)
{
    std::ostream::sentry const  out_sentry(out_stream);

    if (out_sentry) {
        boost::io::ios_all_saver saved_state(out_stream);

        out_stream << std::fixed << std::setprecision(3);

        out_stream << "(";
        for (unsigned r = 0; r < row_dim; ++r) {
            for (unsigned c = 0; c < col_dim; ++c) {
                unsigned out_index = r + c * row_dim;
                out_stream << (out_index != 0 ? " " : "") << out_mat.data_array[out_index];
            }
            if (r != row_dim - 1) out_stream << std::endl;
        }
        out_stream << ")";
    }
    else {
        out_stream.setstate(std::ios_base::failbit);
    }

    return (out_stream);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
std::istream& operator>>(std::istream& in_stream, mat<scal_type, row_dim, col_dim>& in_mat)
{
    std::istream::sentry const  in_sentry(in_stream);

    if (in_sentry) {
        mat<scal_type, row_dim, col_dim>    tmp_mat;
        std::istream::char_type             cur_char(0);
        bool                                bracket_version(false);

        in_stream >> cur_char;

        bracket_version = (cur_char == std::istream::char_type('('));

        if (!bracket_version) {
            in_stream.putback(cur_char);
        }
        for (unsigned r = 0; r < row_dim; ++r) {
            for (unsigned c = 0; c < col_dim; ++c) {
                unsigned in_index = r + c * row_dim;
                in_stream >> tmp_mat.data_array[in_index];
            }
        }
        if (bracket_version) {
            in_stream >> cur_char;
            if (cur_char != std::istream::char_type(')')) {
                in_stream.clear(std::ios_base::badbit);
            }
        }

        if (in_stream) {
            in_mat = tmp_mat;
        }
    }
    else {
        in_stream.setstate(std::ios_base::failbit);
    }

    return (in_stream);
}

} // namespace math
} // namespace scm


#endif // MATH_MAT_STREAM_IO_H_INCLUDED
