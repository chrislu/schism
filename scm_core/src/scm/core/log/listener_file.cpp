
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "listener_file.h"

#include <scm/core/log/message.h>

namespace scm {
namespace log {

listener_file::listener_file(const std::string& file_name, bool append)
  : _file_name(file_name),
    _file_stream(file_name.c_str(), append ? std::ios_base::out | std::ios_base::app : std::ios_base::out | std::ios_base::trunc)
{
    if (!_file_stream) {
        throw std::ios::failure("listener_file::listener_file(): <error> unable to open file: " + file_name);
    }
}

listener_file::~listener_file()
{
}

void
listener_file::notify(const message& msg)
{
    _file_stream << get_log_message(msg);
    _file_stream.flush();
}

} // namespace log
} // namespace scm
