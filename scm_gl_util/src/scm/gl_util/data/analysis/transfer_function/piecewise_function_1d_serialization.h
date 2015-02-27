
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_PIECEWISE_FUNCTION_1D_SERIALIZATION_H_INCLUDED
#define SCM_GL_UTIL_PIECEWISE_FUNCTION_1D_SERIALIZATION_H_INCLUDED

#include <boost/format.hpp>

#include <boost/spirit/include/classic.hpp>
#include <boost/io/ios_state.hpp>

#include <string>
//#include <istream>
//#include <iterator>

#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_1d.h>

namespace scm {
namespace data {

template<typename   val_type,
         typename   res_type>
std::ostream& operator<<(std::ostream& out_stream,
                         const piecewise_function_1d<val_type, res_type>& pw_function)
{
    std::ostream::sentry        out_sentry(out_stream);

    if (!out_sentry) {
        out_stream.setstate(std::ios_base::failbit);
        return (out_stream);
    }

    using boost::format;

    typedef typename piecewise_function_1d<val_type, res_type>::const_stop_iterator iter;

    val_type    vt = val_type(0);
    res_type    rt = res_type(0);

    out_stream << "[piecewise function 1D]" << std::endl;
    out_stream << format("num_stops %10t= %1%\n")                % pw_function.num_stops();
    out_stream << format("stop type_ids %20t= <first  = %1%,\n") % typeid(vt).name();
    out_stream << format("%23tsecond = %1% >\n")                 % typeid(rt).name();

    out_stream << std::endl << "[stops]" << std::endl;
    for (iter i = pw_function.stops_begin(); i != pw_function.stops_end(); ++i) {
        if (i != pw_function.stops_begin())
            out_stream << std::endl;
        out_stream << format("%1% %20t= %2%") % i->first % i->second;
    }

    return (out_stream);
}

// grammer for parsing of output
struct piecewise_function_1d_grammar : public boost::spirit::classic::grammar<piecewise_function_1d_grammar>
{
    piecewise_function_1d_grammar(unsigned& n)
        : num_stops(n){}

    template <typename ScannerT>
    struct definition
    {
        definition(const piecewise_function_1d_grammar& self)
        {
            using namespace boost::spirit::classic;

            header      =      lexeme_d[str_p("[piecewise function 1D]")]
                            >> str_p("num_stops")               >> ch_p('=') >> uint_p[assign_a(self.num_stops)]
                            >> lexeme_d[str_p("stop type_ids")] >> ch_p('=')
                            >> lexeme_d[str_p("<first  =")]     >> lexeme_d[str_p(self.vt_name.c_str())] >> ch_p(',')
                            >> str_p("second")                  >> ch_p('=') >> lexeme_d[str_p(self.rt_name.c_str())] >> ch_p('>');

            stops       = str_p("[stops]");

            expression  = header >> stops;
        }

        boost::spirit::classic::rule<ScannerT> expression, header, stops;

        boost::spirit::classic::rule<ScannerT> const&
        start() const { return expression; }
    };

    unsigned&           num_stops;

    std::string         vt_name;
    std::string         rt_name;
};

template<typename   val_type,
         typename   res_type>
std::istream& operator>>(std::istream& in_stream,
                         piecewise_function_1d<val_type, res_type>& pw_function)
{
    std::istream::sentry        in_sentry(in_stream);

    if (!in_sentry) {
        in_stream.setstate(std::ios_base::failbit);
        return (in_stream);
    }

    using namespace boost::spirit::classic;

    typedef std::istream_iterator<std::istream::char_type>     iterator_t;

    const val_type      vt          = val_type(0);
    const res_type      rt          = res_type(0);

    unsigned            num_stops;

    piecewise_function_1d_grammar gram(num_stops);

    gram.vt_name = typeid(vt).name();
    gram.rt_name = typeid(rt).name();

    boost::io::ios_flags_saver      in_flags_sav(in_stream);

    // turn of white space skipping for parsing
    in_stream.unsetf(std::ios::skipws); 

    iterator_t start(in_stream);
    iterator_t end = iterator_t();
   
    parse_info<iterator_t> info = parse(start, end, gram, space_p); 

    if (!info.hit) {
        in_stream.clear(std::ios_base::badbit);
        return (in_stream);
    }

    in_flags_sav.restore();

    typename piecewise_function_1d<val_type, res_type>::stop_type    tmp_stop;
    piecewise_function_1d<val_type, res_type>                        tmp_func;
    std::istream::char_type                                          tmp_char;

    while (in_stream && !in_stream.eof()) {
        in_stream >> tmp_stop.first;
        in_stream >> tmp_char;
        in_stream >> tmp_stop.second;

        if (!in_stream || tmp_char != std::istream::char_type('=')) {
            in_stream.clear(std::ios_base::badbit);
        }
        else {
            tmp_func.add_stop(tmp_stop);
        }

        in_stream >> std::ws;
    }

    if (in_stream) {
        pw_function = tmp_func;
    }

    return (in_stream);
}

} // namespace data
} // namespace scm

#endif // SCM_GL_UTIL_PIECEWISE_FUNCTION_1D_SERIALIZATION_H_INCLUDED
