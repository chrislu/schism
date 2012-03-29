
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "message.h"

#include <algorithm>
#include <sstream>

#include <scm/time.h>

namespace scm {
namespace log {

message::message(const logger_type& ref_log, const level& lev, const string_type& msg)
  : _sending_logger(ref_log),
    _log_level(lev),
    _message(msg)
{
    _date   = time::universal_date();
    _time   = time::universal_time();
}

message::~message()
{
}

const message::logger_type&
message::sending_logger() const
{
    return _sending_logger;
}

const level&
message::log_level() const
{
    return _log_level;
}

const message::string_type&
message::raw_message() const
{
    return _message;
}

void
message::decorate_message(const string_type& decoration,
                          const string_type& in_message,
                                string_type& out_message) const
{
    ostream_type    predecorated_message;
    ostream_type    decorated_message;
    scm::size_t     decoration_indent = decoration.size();
    stream_type     raw_msg_stream(in_message);
    string_type     raw_msg_line;
    bool            indent_decoration = false;
    scm::size_t     log_indention_width = sending_logger().indent_level() * sending_logger().indent_width();

    while (std::getline(raw_msg_stream, raw_msg_line)) {
        if (   (0 < decoration_indent)
            && indent_decoration
            && !raw_msg_line.empty())
        {
            std::fill_n(std::ostreambuf_iterator<char_type>(predecorated_message.rdbuf()),
                        decoration_indent,
                        char_type(' '));
        }
        if (  (0 < log_indention_width)
            && !raw_msg_line.empty())
        {
            std::fill_n(std::ostreambuf_iterator<char_type>(predecorated_message.rdbuf()),
                        log_indention_width,
                        sending_logger().indent_fill_char());
        }
        predecorated_message << raw_msg_line << std::endl;
        indent_decoration = true;
    }
    _postdec_message = predecorated_message.str();
    decorated_message << decoration << predecorated_message.str();
    decorated_message.str().swap(out_message);
    //out_message.swap(decorated_message.str());
}

const message::string_type&
message::plain_message() const
{
    if (_plain_message.empty()) {
        decorate_message("", raw_message(), _plain_message);
    }
    _postdec_decoration.clear();

    return _plain_message;
}

const message::string_type&
message::decorated_message() const
{
    if (_decorated_message.empty()) {
        ostream_type msg_decoration;

        if (!sending_logger().name().empty()) {
            msg_decoration << sending_logger().name() << " ";
        }
        msg_decoration << "<" << log_level().to_string() << "> ";

        decorate_message(msg_decoration.str(), raw_message(), _decorated_message);
        msg_decoration.str().swap(_postdec_decoration);
    }

    return _decorated_message;
}

const message::string_type&
message::full_decorated_message() const
{
    if (_full_decorated_message.empty()) {
        ostream_type msg_decoration;

        msg_decoration << _time << ": ";
        if (!sending_logger().name().empty()) {
            msg_decoration << sending_logger().name() << " ";
        }
        msg_decoration << "<" << log_level().to_string() << "> ";

        decorate_message(msg_decoration.str(), raw_message(), _full_decorated_message);
        msg_decoration.str().swap(_postdec_decoration);
    }

    return _full_decorated_message;
}

const message::string_type&
message::postdec_decoration() const
{
    return _postdec_decoration;
}

const message::string_type&
message::postdec_message() const
{
    return _postdec_message;
}

} // namespace log
} // namespace scm
