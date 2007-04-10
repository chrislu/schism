
#include "console.h"

using namespace scm::core;

#pragma todo(make _input_history_max_length variable)

console_interface::console_interface(input_history_container::size_type hs)
    : /*_output_buffer(new std::ostringstream()),
      _input_buffer(new std::stringstream()),*/
      _input_history_max_length(hs) 
{
}

console_interface::~console_interface()
{
}

void console_interface::add_input(const std::string& inp)
{
    _input_buffer << inp;
}

bool console_interface::process_input()
{
    add_input_history(_input_buffer.str());
    _input_buffer.clear();

     //script_system::do_string()
     //maybe _output_buffer << script_system::get_result_string();

    return (true);
}

void console_interface::add_input_history(const std::string& inp)
{
    _input_history.push_back(inp);

    while (_input_history.size() > _input_history_max_length) {
        _input_history.pop_front();
    }
}
