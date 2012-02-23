
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_LOG_STREAMBUF_H_INCLUDED
#define SCM_CORE_LOG_LOG_STREAMBUF_H_INCLUDED

#include <streambuf> 
#include <algorithm> 
#include <iterator> 
#include <iosfwd>

#include <scm/core/memory.h>

namespace scm {
namespace string {

class log_streambuf_base
{
public:
    typedef shared_ptr<log_streambuf_base>  log_streambuf_pointer;

public:
    log_streambuf_base() {};
    virtual ~log_streambuf_base() {};

    //virtual log_streambuf_pointer clone() const = 0;

}; // class log_streambuf_base


template <typename char_type,
          typename traits = std::char_traits<char_type> >
struct basic_log_streambuf : public log_streambuf_base,
                             public std::basic_streambuf<char_type,traits> 
{
public:
	typedef typename traits::int_type int_type;
	typedef typename traits::pos_type pos_type;
	typedef typename traits::off_type off_type;

private:
    typedef std::basic_streambuf<char_type, traits>     streambuf_type;

public:
    basic_log_streambuf(streambuf_type* sbuf)
      : _streambuf(sbuf),
        _fill_char(' '),
        _indent_count(0),
        _indent_width(4),
        _set_indent(true)
    {
    }
    virtual ~basic_log_streambuf()
    {
        sync();
    }

    //log_streambuf_pointer clone() const {
    //    return make_shared<basic_log_streambuf<char_type, traits> >(*this);
    //}

    void copy_indention_attributes(const basic_log_streambuf<char_type, traits>& rhs) {
        _fill_char      = rhs._fill_char;
        _indent_count   = rhs._indent_count;
        _indent_width   = rhs._indent_width;
        _set_indent     = rhs._set_indent;
    }

    char_type fill_char() const {
        return (_fill_char);
    }
    void fill_char(char_type c) {
        _fill_char = c;
    }

    int indention() const {
        return _indent_count;
    }
    void indention(int i) {
        _indent_count = i;
    }

    int indention_width() const {
        return _indent_width;
    }
    void indention_width(int i) {
        _indent_width = i;
    }



    streambuf_type* original_rdbuf() {
        return (_streambuf);
    }

private:
    basic_log_streambuf(const basic_log_streambuf<char_type, traits>& rhs)
      : _streambuf(rhs._streambuf),
        _fill_char(rhs._fill_char),
        _indent_count(rhs._indent_count),
        _indent_width(rhs._indent_width),
        _set_indent(rhs._set_indent)
    {
    }

    // declared, never defined
    basic_log_streambuf<char_type, traits>& operator=(const basic_log_streambuf<char_type, traits>& );
    
    // override std::streambuf::overflow
    virtual int_type overflow(int_type c = traits::eof())
    {
        if (traits::eq_int_type(c, traits::eof())) {
            return _streambuf->sputc(static_cast<char_type>(c));
        }
        if (_set_indent) {
            std::fill_n(std::ostreambuf_iterator<char_type>(_streambuf), _indent_count * _indent_width, _fill_char);
            _set_indent = false;
        }
        if (traits::eq_int_type(_streambuf->sputc(static_cast<char_type>(c)), traits::eof())) {
            return traits::eof();
        }
        if (traits::eq_int_type(c, traits::to_char_type('\n'))) {
            _set_indent = true;
        }
        return traits::not_eof(c);
    }

private:
    streambuf_type* _streambuf;
    
    char_type       _fill_char;
    int             _indent_count;
    int             _indent_width;
    bool            _set_indent; 

    // log_level
    // logger
    // -> sync

}; // class basic_log_streambuf

typedef basic_log_streambuf<char>       log_streambuf;
typedef basic_log_streambuf<wchar_t>    wlog_streambuf;

} // namespace string
} // namespace scm

#endif // SCM_CORE_LOG_LOG_STREAMBUF_H_INCLUDED
