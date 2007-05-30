
#include "console_output_listener.h"

#include <boost/bind.hpp>

#include <scm_core/console/console_system.h>

using namespace scm::con;

console_output_listener::console_output_listener(console_system& con)
    : _console(con),
      _log_threshold(con::warning)
{
    _connection = _console._out_stream.connect_output_listener(boost::bind(&console_output_listener::update, this, _1, _2));
}

console_output_listener::~console_output_listener()
{
    _console._out_stream.disconnect_output_listener(_connection);
}

void console_output_listener::set_log_threshold(int threshold)
{
    _log_threshold = threshold;
}
