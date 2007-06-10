
#include "console_output_listener.h"

#include <boost/bind.hpp>

using namespace scm::con;

console_output_listener::console_output_listener()
  : _log_threshold(con::info)
{
}

console_output_listener::~console_output_listener()
{
    disconnect();
}

bool console_output_listener::connect(console_out_stream& con)
{
    _connection = con.connect_output_listener(boost::bind(&console_output_listener::update, this, _1, _2));

    return (_connection.connected());
}

bool console_output_listener::disconnect()
{
    _connection.disconnect();

    return (!_connection.connected());
}

void console_output_listener::set_log_threshold(int threshold)
{
    _log_threshold = threshold;
}
