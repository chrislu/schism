
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "out_stream_manip.h"

#include <scm/core/log/out_stream.h>

namespace scm {
namespace log {

out_stream&
trace(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_trace)); 
    return os;
}

out_stream&
debug(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_debug)); 
    return os;
}

out_stream&
info(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_info)); 
    return os;
}

out_stream&
output(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_output)); 
    return os;
}

out_stream&
warning(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_warning)); 
    return os;
}

out_stream&
error(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_error)); 
    return os;
}

out_stream&
fatal(out_stream& os)
{
    os.switch_log_level(level(scm::log::ll_fatal)); 
    return os;
}

out_stream&
nline(out_stream& os)
{
    if (os.log_level() <= os.associated_logger().log_level()) {
        out_stream::ostream_type& oss = os.ostream();
        oss.put(oss.widen('\n'));
    }
    return os;
}

out_stream& end(out_stream& os)
{
    if (os.log_level() <= os.associated_logger().log_level()) {
        os.ostream() << std::endl; 
        os.flush();
    }
    return os;
}

out_stream& flush(out_stream& os)
{
    os.flush();
    return os;
}

out_stream& indent(out_stream& os)
{
    os.flush();
    os.associated_logger().increase_indent_level();

    return os;
}

out_stream& outdent(out_stream& os)
{
    os.flush();
    os.associated_logger().decrease_indent_level();

    return os;
}

indent_fill::indent_fill(out_stream::char_type c)
  : _c(c)
{
}

indent_fill::~indent_fill()
{
}

out_stream&
indent_fill::do_manip(out_stream& los) const
{
    los.associated_logger().indent_fill_char(_c);
    return (los);
}

indent_width::indent_width(int w)
  : _w(w)
{
}

indent_width::~indent_width()
{
}

out_stream&
indent_width::do_manip(out_stream& los) const
{
    los.associated_logger().indent_width(_w);
    return (los);
}

} // namespace log
} // namespace scm
