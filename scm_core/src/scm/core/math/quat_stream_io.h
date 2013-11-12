
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_QUAT_STREAM_IO_H_INCLUDED
#define MATH_QUAT_STREAM_IO_H_INCLUDED

#include <iomanip>
#include <ostream>
#include <istream>
#include <boost/io/ios_state.hpp>

namespace scm {
namespace math {

template<typename scal_type>
std::ostream& operator<<(std::ostream& out_stream, const quat<scal_type>& out_quat)
{
    std::ostream::sentry const  out_sentry(out_stream);

    if (out_sentry) {
        boost::io::ios_all_saver saved_state(out_stream);

        out_stream << std::fixed << std::setprecision(3);

        out_stream << "("
                   << out_quat.w << "  "
                   << out_quat.i << "  "
                   << out_quat.j << "  "
                   << out_quat.k << ")";
    }
    else {
        out_stream.setstate(std::ios_base::failbit);
    }

    return (out_stream);
}

template<typename scal_type>
std::istream& operator>>(std::istream& in_stream, quat<scal_type>& in_quat)
{
    std::istream::sentry const  in_sentry(in_stream);

    if (in_sentry) {
        quat<scal_type>             tmp_quat;
        std::istream::char_type     cur_char(0);
        bool                        bracket_version(false);

        in_stream >> cur_char;

        bracket_version = (cur_char == std::istream::char_type('('));

        if (!bracket_version) {
            in_stream.putback(cur_char);
        }

        in_stream >> tmp_quat.w >> tmp_quat.i >> tmp_quat.j >> tmp_quat.k;

        if (bracket_version) {
            in_stream >> cur_char;
            if (cur_char != std::istream::char_type(')')) {
                in_stream.clear(std::ios_base::badbit);
            }
        }

        if (in_stream) {
            in_quat = tmp_quat;
        }
    }
    else {
        in_stream.setstate(std::ios_base::failbit);
    }

    return (in_stream);
}

} // namespace math
} // namespace scm

#endif // MATH_QUAT_STREAM_IO_H_INCLUDED
