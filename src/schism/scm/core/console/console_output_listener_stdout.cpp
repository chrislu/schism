
#include "console_output_listener_stdout.h"

#include <iostream>
#include <string>

using namespace scm::con;

console_output_listener_stdout::console_output_listener_stdout(console_system& con)
    : console_output_listener(con)
{
}

console_output_listener_stdout::~console_output_listener_stdout()
{
}

void console_output_listener_stdout::update(const std::string&          update_buffer,
                                            const console_out_stream&   stream_source)
{
    if (_log_threshold >= stream_source.get_log_level()) {
        std::cout << update_buffer;
    }
}
