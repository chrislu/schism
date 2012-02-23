
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "logger_state.h"

#include <scm/core/log/out_stream.h>

namespace scm {
namespace log {

logger_format_saver::logger_format_saver(logger_type& l)
  : _logger(l),
    _s_indent_fill_char(l.indent_fill_char()),
    _s_indent_level(l.indent_level()),
    _s_indent_width(l.indent_width())
{
}

//logger_format_saver::logger_format_saver(out_stream& los)
//  : _logger(los.associated_logger()),
//    _s_indent_fill_char(los.associated_logger().indent_fill_char()),
//    _s_indent_level(los.associated_logger().indent_level()),
//    _s_indent_width(los.associated_logger().indent_width())
//{
//}

logger_format_saver::~logger_format_saver()
{
    restore();
}

void
logger_format_saver::restore()
{
    _logger.indent_fill_char(_s_indent_fill_char);
    _logger.indent_level(_s_indent_level);
    _logger.indent_width(_s_indent_width);
}

} // namespace log
} // namespace scm
