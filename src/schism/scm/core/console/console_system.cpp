
#include "console_system.h"

#include <scm/console.h>

using namespace scm::con;

#pragma todo(make _input_history_max_length variable)

console_system::console_system(input_history_container::size_type hs)
    : _input_history_max_length(hs) 
{
}

console_system::~console_system()
{
}

bool console_system::initialize()
{
    if (_initialized) {
        console.get() << con::log_level(con::warning)
                      << "console_system::initialize(): "
                      << "allready initialized" << std::endl;
        return (true);
    }

    _initialized = true;
    return (true);
}

bool console_system::shutdown()
{
    _initialized = false;
    return (true);
}

void console_system::add_input(const std::string& inp)
{
    _input_buffer << inp;
}

bool console_system::process_input()
{
    add_input_history(_input_buffer.str());
    _input_buffer.clear();

     //script_system::do_string()
     //maybe _output_buffer << script_system::get_result_string();

    return (true);
}

void console_system::add_input_history(const std::string& inp)
{
    _input_history.push_back(inp);

    while (_input_history.size() > _input_history_max_length) {
        _input_history.pop_front();
    }
}
